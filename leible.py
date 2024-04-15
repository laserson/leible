import os
import email
import email.policy
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

import pandas as pd
import requests
import sqlite_utils
import typer
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from ratelimit import limits, sleep_and_retry
from rich import print
from tqdm import tqdm
from imapclient import IMAPClient


######################
#   Database stuff   #
######################


@dataclass
class Article:
    title: str
    abstract: str
    journal: str = None
    year: int = None
    # one of these ids should be set
    doi: str = None
    pmid: str = None
    arxiv: str = None
    # internal database id
    _id: int = None
    # mainly an analysis convenience
    source: str = None


def tracer(sql, params):
    logger.debug("SQL: {}\nPARAMS: {}", sql, params)


def get_leible_db(db_path: str) -> sqlite_utils.Database:
    db = sqlite_utils.Database(db_path, tracer=tracer)
    db["articles"].create(
        Article.__annotations__,
        pk="_id",
        if_not_exists=True,
    )
    return db


def is_article_in_db(db: sqlite_utils.Database, article: Article) -> bool:
    if article.pmid is not None and db["articles"].count_where("pmid = ?", [article.pmid]) > 0:
        return True
    if article.doi is not None and db["articles"].count_where("doi = ?", [article.doi]) > 0:
        return True
    if article.arxiv is not None and db["articles"].count_where("arxiv = ?", [article.arxiv]) > 0:
        return True
    return False


def insert_article(db: sqlite_utils.Database, article: Article) -> None:
    if article.doi is None and article.pmid is None and article.arxiv is None:
        raise ValueError("Article must have a DOI, PMID, or arXiv ID")
    if article.title is None or article.abstract is None:
        raise ValueError("Article must have a title and abstract")
    db["articles"].insert(asdict(article))
    logger.debug("Inserted article: {}", article.title)


########################
#   Processing Email   #
########################


def fetch_emails(
    imap_host, imap_port, imap_user, imap_pass, folder="ToC"
) -> list[email.message.EmailMessage]:
    with IMAPClient(host=imap_host, port=imap_port, use_uid=True) as client:
        client.login(imap_user, imap_pass)
        client.select_folder(folder, readonly=True)
        message_ids = client.search()
        emails = []
        for message_id, message_data in client.fetch(message_ids, [b"RFC822"]).items():
            message = email.message_from_bytes(
                message_data[b"RFC822"], policy=email.policy.default
            )
            emails.append(message)
        return emails


def extract_html_from_email(message: email.message.EmailMessage) -> BeautifulSoup:
    return BeautifulSoup(
        message.get_body(("html", "plain")).get_payload(), "html.parser"
    )


def parse_email_biorxiv(message: email.message.EmailMessage) -> list[Article]:
    subject = message.get("Subject")
    source = f"BioRxiv {subject}"
    soup = extract_html_from_email(message)
    articles = []
    for elt in soup.find_all("a"):
        url = elt.get("href")
        if url.startswith(
            "http://www.biorxiv.org/cgi/content/abstract"
        ) or url.startswith("http://www.biorxiv.org/content/early"):
            article = get_article_from_biorxiv(url)
            article.source = source
            articles.append(article)
    return articles


def parse_email_google_scholar(message: email.message.EmailMessage) -> list[str]:
    pass


def parse_email_pubmed(message: email.message.EmailMessage) -> list[str]:
    pass


def parse_email_nature(message: email.message.EmailMessage) -> list[str]:
    pass


def parse_email_science(message: email.message.EmailMessage) -> list[str]:
    pass


def parse_email_cellpress(message: email.message.EmailMessage) -> list[str]:
    pass


def parse_email_nejm(message: email.message.EmailMessage) -> list[str]:
    pass


def parse_email_annual_reviews(message: email.message.EmailMessage) -> list[str]:
    pass


def parse_email_oxford_academic(message: email.message.EmailMessage) -> list[str]:
    pass


def parse_email_biodecoded(message: email.message.EmailMessage) -> list[str]:
    pass


def parse_email(message: email.message.EmailMessage) -> list[Article]:
    from_ = message.get("From")
    to = message.get("To")
    subject = message.get("Subject")

    if from_ == "cshljnls-mailer@alerts.highwire.org":
        # BioRxiv
        return parse_email_biorxiv(message)
    if from_ == "scholaralerts-noreply@google.com":
        # Google Scholar
        logger.debug("NOT IMPLEMENTED Google Scholar")
    if from_ == "scholarcitations-noreply@google.com":
        # Google Scholar Citations
        logger.debug("NOT IMPLEMENTED Google Scholar")
    if from_ == "efback@ncbi.nlm.nih.gov":
        # PubMed
        logger.debug("NOT IMPLEMENTED PubMed")
    if from_ == "ealert@nature.com":
        # Nature ToC
        logger.debug("NOT IMPLEMENTED Nature")
    if from_ == "oxfordacademicalerts@oup.com":
        # Oxford Academic (e.g., NAR)
        logger.debug("NOT IMPLEMENTED Oxford Academic")
    if from_ == "alerts@aaas.sciencepubs.org":
        # Science or STM
        logger.debug("NOT IMPLEMENTED Science")
    if from_ == "cellpress@notification.elsevier.com":
        # Cell Press
        logger.debug("NOT IMPLEMENTED Cell Press")
    if from_ == "nejmtoc@n.nejm.org":
        # NEJM
        logger.debug("NOT IMPLEMENTED NEJM")
    if from_ == "busybee@blogtrottr.com" and "BioDecoded" in subject:
        # BioDecoded
        logger.debug("NOT IMPLEMENTED BioDecoded")
    if from_ == "announce@annualreviews.org":
        # Annual Reviews
        logger.debug("NOT IMPLEMENTED Annual Reviews")
    return []


########################
#   Article metadata   #
########################


@sleep_and_retry
@limits(calls=1, period=1)
def get_article_from_biorxiv(url: str) -> Article:
    response = requests.get(url)
    logger.debug("BioRxiv STATUS {} GET {}", response.status_code, response.url)
    if response.status_code != 200:
        raise ValueError(f"Failed to load article: {response.url}")
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("h1", attrs={"id": "page-title"}).get_text("\n")
    abstract = soup.find("div", attrs={"class": "abstract"}).get_text("\n")
    journal = "BioRxiv"
    year = int(
        soup.find("meta", attrs={"name": "article:published_time"})
        .get("content")
        .split("-")[0]
    )
    doi = soup.find("meta", attrs={"name": "citation_doi"}).get("content")
    return Article(title=title, abstract=abstract, journal=journal, year=year, doi=doi)


@sleep_and_retry
@limits(calls=1, period=1)
def get_article_by_pmid(pmid) -> Article:
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "tool": "leible",
            "email": "u@snkt.st",
        },
    )
    logger.debug("PMID {} STATUS {} GET {}", pmid, response.status_code, response.url)
    if response.status_code != 200:
        raise ValueError(f"Failed to load article: {response.url}")
    xml_data = response.text
    soup = BeautifulSoup(xml_data, "xml")
    title = soup.find("ArticleTitle").text.strip()
    abstract = soup.find("AbstractText").text.strip()
    journal = soup.find("Journal").find("Title").text.strip()
    year = int(soup.find("PubDate").find("Year").text.strip())
    doi = (
        soup.find("ArticleIdList")
        .find("ArticleId", attrs={"IdType": "doi"})
        .get_text()
        .strip()
    )
    return Article(
        title=title, abstract=abstract, journal=journal, year=year, doi=doi, pmid=pmid
    )


def doi_to_url(doi: str) -> str:
    response = requests.head(f"https://doi.org/{doi}", allow_redirects=True)
    return response.url


##################
#   Embeddings   #
##################


def extract_article_text(article: Article) -> str:
    return f"{article.title}\n\n{article.abstract}"


@sleep_and_retry
@limits(calls=5, period=1)
def get_article_embedding(client: OpenAI, model: str, article: Article) -> list[float]:
    input = extract_article_text(article)
    response = client.embeddings.create(input=input, model=model)
    return response.data[0].embedding


###########
#   CLI   #
###########


app = typer.Typer()


@app.command(name="load-csv")
def load_csv(db_path: str, input_csv: str):
    """Load articles from a CSV into the database.

    Articles are fetched from these columns in this order of priority:
    - `pmid`
    - `doi`

    Articles are added to the database if they can be matched and have a Title and Abstract.
    """
    db = get_leible_db(db_path)
    table = db["articles"]
    input_df = pd.read_csv(input_csv, dtype=str)

    num_inserted = 0
    for i, record in tqdm(enumerate(input_df.itertuples()), total=len(input_df)):
        msg = f"{i+1:>6}  "
        if pd.notna(record.pmid):
            msg += f"PMID: {record.pmid:<12}  "
            if table.count_where("pmid = ?", [record.pmid]) > 0:
                msg += "SKIP Already in database"
                print(msg)
                continue
            try:
                article = get_article_by_pmid(record.pmid)
            except Exception as e:
                print("MODIFY EXPECTION TYPE IN CODE")
                msg += "SKIP No abstract"
                print(msg)
                continue
            msg += f"Title: {article.title}"
            table.insert(article.asdict())
            num_inserted += 1
        elif pd.notna(record.doi):
            msg += f"DOI: {record.doi:<24}  "
            pass
        else:
            msg += "SKIP no pmid no doi"
        print(msg)
    print(f"Inserted {num_inserted} new records out of {len(input_df)} processed.")


@app.command()
def load_imap(db_path: str):
    """Load articles from an IMAP mailbox into the database."""

    load_dotenv()

    emails = fetch_emails(
        os.environ["IMAP_SERVER"],
        os.environ["IMAP_PORT"],
        os.environ["IMAP_USER"],
        os.environ["IMAP_PASS"],
        os.environ["IMAP_FOLDER"],
    )

    db = get_leible_db(db_path)

    num_inserted = 0
    num_processed = 0
    for message in tqdm(emails):
        articles = parse_email(message)
        for article in tqdm(articles):
            num_processed += 1
            if is_article_in_db(db, article):
                logger.debug("Article already in database: {}", article.title)
                continue
            insert_article(db, article)
            num_inserted += 1
    print(f"Inserted {num_inserted} new records out of {num_processed} processed.")


@app.command()
def embed(db_path: str, model: str = "text-embedding-3-small"):
    """Embed articles in the database using the specified model."""
    load_dotenv()
    client = OpenAI()

    db = sqlite_utils.Database(db_path, tracer=tracer)
    table = db.table("articles")
    table.add_column(model, "BLOB")
    for article in tqdm(table.rows_where("`text-embedding-3-small` is null")):
        embedding = get_article_embedding(
            client, model, article["title"], article["abstract"]
        )
        table.update(article["id"], {model: embedding})


@app.command()
def cluster(db_path: str):
    pass


if __name__ == "__main__":
    import sys

    logger.remove()
    # logger.add(sys.stderr, level="INFO")
    logger.add(sys.stderr, level="DEBUG")

    app()
