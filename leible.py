import email
import email.policy
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import sqlite_utils
import torch
import typer
from adapters import AutoAdapterModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from imapclient import IMAPClient
from joblib import Parallel, delayed
from loguru import logger
from openai import OpenAI
from ratelimit import limits, sleep_and_retry
from rich import print
from semanticscholar import SemanticScholar
from toolz import partition_all
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

######################
#   Database stuff   #
######################


@dataclass
class Article:
    title: str
    abstract: str
    journal: str = None
    year: int = None
    source: str = None  # e.g., BioRxiv email, Papers app
    date_added: str = None  # YYYY-MM-DD
    notes: str = None  # extra info, preferably in JSON
    # one of these ids _should_ be set
    doi: str = None
    pmid: str = None
    arxiv: str = None
    # internal database id, set when inserted into the database
    _id: int = None


@dataclass
class Embedding:
    input: str  # the input text that was embedded
    model: str  # the name of the model
    embedding: str  # JSON list of floats (as string)
    article_id: int = None  # foreign key to articles _id


def get_or_create_leible_db(db_path: str | None = None) -> sqlite_utils.Database:
    db = sqlite_utils.Database(db_path or ":memory:")
    db["articles"].create(
        Article.__annotations__,
        pk="_id",
        if_not_exists=True,
    )
    db["embeddings"].create(
        Embedding.__annotations__,
        pk=("article_id", "model"),
        foreign_keys=[("article_id", "articles", "_id")],
        if_not_exists=True,
    )
    return db


def does_article_match(db: sqlite_utils.Database, article: Article) -> bool:
    for field in ["pmid", "doi", "arxiv"]:
        value = getattr(article, field)
        if (
            value is not None
            and db["articles"].count_where(f"{field} = ?", [value]) > 0
        ):
            return True
    return False


def insert_article(db: sqlite_utils.Database, article: Article) -> int:
    article.date_added = datetime.now().strftime("%Y-%m-%d")
    return db["articles"].insert(asdict(article)).last_pk


def db_has_value(db: sqlite_utils.Database, column: str, value: str) -> bool:
    return db["articles"].count_where(f"{column} = ?", [value]) > 0


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
            try:
                article = get_article_from_biorxiv(url)
            except Exception as e:
                logger.error("Failed to load article: {}", url)
                continue
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
    response.raise_for_status()
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


def extract_article_from_pubmed_xml(soup: BeautifulSoup) -> Article:
    try:
        title = soup.find("ArticleTitle").text.strip()
    except AttributeError:
        raise ValueError("No title found for this article")

    try:
        abstract = soup.find("AbstractText").text.strip()
    except AttributeError:
        raise ValueError("No abstract found for this article")

    try:
        journal = soup.find("Journal").find("Title").text.strip()
    except AttributeError:
        journal = None

    try:
        year = int(soup.find("PubDate").find("Year").text.strip())
    except AttributeError:
        year = None

    try:
        doi = (
            soup.find("ArticleIdList")
            .find("ArticleId", attrs={"IdType": "doi"})
            .text.strip()
        )
    except AttributeError:
        doi = None

    try:
        pmid = (
            soup.find("ArticleIdList")
            .find("ArticleId", attrs={"IdType": "pubmed"})
            .text.strip()
        )
    except AttributeError:
        pmid = None

    return Article(
        title=title, abstract=abstract, journal=journal, year=year, doi=doi, pmid=pmid
    )


@sleep_and_retry
@limits(calls=1, period=1)
def get_articles_by_pmids(pmids: list[str]) -> list[Article]:
    response = requests.post(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        data={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": "leible",
            "email": "u@snkt.st",
        },
    )
    response.raise_for_status()
    articles = []
    for article_soup in BeautifulSoup(response.text, "xml").find_all("PubmedArticle"):
        try:
            article = extract_article_from_pubmed_xml(article_soup)
        except ValueError as e:
            logger.debug("Article missing title or abstract: {}\n{}", e, article_soup)
            continue
        articles.append(article)
    return articles


def get_article_by_pmid(pmid: str) -> Article:
    articles = get_articles_by_pmids([pmid])
    assert len(articles) == 1
    return articles[0]


@sleep_and_retry
@limits(calls=5, period=1)
def doi_to_redirected_url(doi: str) -> str:
    response = requests.head(f"https://doi.org/{doi}", allow_redirects=True)
    return response.url


##################
#   Embeddings   #
##################


@sleep_and_retry
@limits(calls=1, period=1)
def embed_articles_openai(
    client: OpenAI, model: str, articles: list[Article]
) -> list[Embedding]:
    assert len(articles) < 2048, "OpenAI API only supports up to 2048 inputs"
    inputs = [f"{article.title}\n\n{article.abstract}" for article in articles]
    response = client.embeddings.create(input=inputs, model=model)
    return [
        Embedding(
            article_id=article._id, input=input, model=model, embedding=data.embedding
        )
        for article, input, data in zip(articles, inputs, response.data)
    ]


def embed_articles_specter2(
    articles: list[Article], n_jobs: int = 4, batch_size: int = 32
) -> list[Embedding]:
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2", source="hf", load_as="specter2", set_active=True
    )

    dataset = [
        f"{article.title}{tokenizer.sep_token}{article.abstract}"
        for article in articles
    ]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def embed_batch(batch):
        with torch.no_grad():
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            )
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            return batch_embeddings

    embeddings = Parallel(n_jobs=n_jobs)(
        delayed(embed_batch)(batch) for batch in dataloader
    )
    embeddings = torch.cat(embeddings, dim=0)
    return [
        Embedding(
            article_id=article._id,
            input=input,
            model="specter2",
            embedding=embedding.tolist(),
        )
        for article, input, embedding in zip(articles, dataset, embeddings)
    ]


def embed_readcube_papers_export(csv_path: str) -> list[Embedding]:
    df = pd.read_csv(csv_path, dtype=str).query(
        "`created (Read-Only)` > '2022-01-01' or year > '2019'"
    )
    dois = df.dropna(subset="doi")["doi"].tolist()
    sch = SemanticScholar()
    papers = []
    missing_ids = []
    for batch in partition_all(500, dois):
        found, missing = sch.get_papers(batch, return_not_found=True)  # network call
        papers.extend(found)
        missing_ids.extend(missing)
    articles = [Article(title=p.title, abstract=p.abstract) for p in papers]
    return embed_articles_specter2(articles)  # compute heavy


# def get_article_ids_missing_embedding(
#     db: sqlite_utils.Database, model: str
# ) -> list[int]:
#     query = """
#         SELECT a.*, e.*
#         FROM articles AS a
#         LEFT OUTER JOIN
#             (SELECT *
#             FROM embeddings
#             WHERE embeddings.model = ?) AS e ON a._id = e.article_id
#         WHERE e.embedding IS NULL
#     """
#     return [row["_id"] for row in db.query(query, [model])]


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
    db = get_or_create_leible_db(db_path)
    input_df = pd.read_csv(input_csv, dtype=str)

    num_inserted = 0
    for i, record in tqdm(enumerate(input_df.itertuples()), total=len(input_df)):
        msg = f"{i+1:>6}  "
        if pd.notna(record.pmid):
            msg += f"PMID: {record.pmid:<12}  "
            try:
                article = get_article_by_pmid(record.pmid)
            except Exception as e:
                print("MODIFY EXCEPTION TYPE IN CODE")
                msg += "SKIP No abstract?"
                print(msg)
                continue
            if does_article_match(db, article):
                msg += "SKIP Already in database"
                print(msg)
                continue
            msg += f"Title: {article.title}"
            insert_article(db, article)
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
    db = get_or_create_leible_db(db_path)

    num_inserted = 0
    num_processed = 0
    for message in tqdm(emails):
        articles = parse_email(message)
        for article in tqdm(articles):
            num_processed += 1
            if does_article_match(db, article):
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
    db = get_or_create_leible_db(db_path)

    for _id in tqdm(get_article_ids_missing_embedding(db, model)):
        article = Article(**db["articles"].get(_id))
        embedding = embed_articles_openai(client, model, [article])
        db["embeddings"].upsert(
            {
                "article_id": article._id,
                "model": model,
                "embedding": embedding.embedding,
            },
            pk=("article_id", "model"),
        )


@app.command()
def cluster(db_path: str):
    pass


if __name__ == "__main__":
    import sys

    logger.remove()
    # logger.add(sys.stderr, level="INFO")
    logger.add(sys.stderr, level="DEBUG")

    app()
