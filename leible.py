import os
import email
import email.policy
from dataclasses import dataclass

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
from rich.progress import track
from imapclient import IMAPClient


@dataclass
class Article:
    pmid: str
    biorxiv: str
    arxiv: str
    doi: str
    journal: str
    title: str
    abstract: str
    year: int


def tracer(sql, params):
    logger.debug(f"SQL: {sql}\nPARAMS: {params}")


def ensure_article_table(db: sqlite_utils.Database):
    db["articles"].create(
        {
            "id": int,
            "pmid": str,
            "title": str,
            "abstract": str,
        },
        pk="id",
        if_not_exists=True,
    )


def fetch_emails(imap_host, imap_port, imap_user, imap_pass):
    with IMAPClient(host=imap_host, port=imap_port, use_uid=True) as client:
        client.login(imap_user, imap_pass)
        client.select_folder("ToC", readonly=True)
        message_ids = client.search()
        emails = []
        for message_id, message_data in client.fetch(
            message_ids, [b"ENVELOPE", b"RFC822"]
        ).items():
            envelope = message_data[b"ENVELOPE"]
            message = email.message_from_bytes(
                message_data[b"RFC822"], policy=email.policy.default
            )
            emails.append((envelope, message))
        return emails


def parse_email(envelope, message):
    from_ = f"{envelope.from_[0].mailbox.decode()}@{envelope.from_[0].host.decode()}"
    subject = envelope.subject.decode()

    if (
        from_ == "cshljnls-mailer@alerts.highwire.org"
        and subject == "bioRxiv Subject Collection Alert"
    ):
        # bioRxiv subject collection
        pass
    if from_ == "cshljnls-mailer@alerts.highwire.org":
        # bioRxiv keyword
        pass
    if from_ == "scholaralerts-noreply@google.com":
        # Google Scholar
        pass
    if from_ == "scholarcitations-noreply@google.com":
        # Google Scholar Citations
        pass
    if from_ == "efback@ncbi.nlm.nih.gov":
        # PubMed
        pass
    if from_ == "ealert@nature.com":
        # Nature ToC
        pass
    if from_ == "oxfordacademicalerts@oup.com":
        # Oxford Academic (e.g., NAR)
        pass
    if from_ == "alerts@aaas.sciencepubs.org":
        # Science or STM
        pass
    if from_ == "cellpress@notification.elsevier.com":
        # Cell Press
        pass
    if from_ == "nejmtoc@n.nejm.org":
        # NEJM
        pass
    if from_ == "busybee@blogtrottr.com" and "BioDecoded" in subject:
        # BioDecoded
        pass
    if from_ == "announce@annualreviews.org":
        # Annual Reviews
        pass


@sleep_and_retry
@limits(calls=1, period=1)
def get_article_by_pmid(pmid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    response = requests.get(
        url,
        params={
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "tool": "leible",
            "email": "u@snkt.st",
        },
    )
    logger.debug(f"PMID {pmid} STATUS {response.status_code} GET {response.url}")
    if response.status_code != 200:
        raise ValueError(f"Failed to download article: {response.url}")
    xml_data = response.text
    soup = BeautifulSoup(xml_data, "xml")
    title = soup.find("ArticleTitle").text.strip()
    abstract = soup.find("AbstractText")
    abstract = abstract.text.strip() if abstract is not None else None
    return title, abstract


@sleep_and_retry
@limits(calls=5, period=1)
def get_article_embedding(client: OpenAI, model: str, title, abstract) -> list[float]:
    input = f"{title}\n\n{abstract}"
    response = client.embeddings.create(input=input, model=model)
    return response.data[0].embedding


app = typer.Typer()


@app.command(name="load-csv")
def load_csv(db_path: str, input_csv: str):
    """Load articles from a CSV into the database.

    Articles are fetched from these columns in this order of priority:
    - `pmid`

    Articles are added to the database if they can be matched and have a Title and Abstract.
    """
    db = sqlite_utils.Database(db_path, tracer=tracer)
    ensure_article_table(db)
    table = db.table("articles")

    input_df = pd.read_csv(input_csv, dtype=str)
    num_inserted = 0
    for i, record in track(enumerate(input_df.itertuples()), total=len(input_df)):
        msg = f"{i+1:>6}  "
        if pd.notna(record.pmid):
            msg += f"PMID: {record.pmid:<12}  "
            if table.count_where("pmid = ?", [record.pmid]) > 0:
                msg += "SKIP Already in database"
                print(msg)
                continue
            title, abstract = get_article_by_pmid(record.pmid)
            if abstract is None:
                msg += "SKIP No abstract"
                print(msg)
                continue
            msg += f"Title: {title}"
            table.insert(
                {
                    "id": None,
                    "pmid": record.pmid,
                    "title": title,
                    "abstract": abstract,
                }
            )
            num_inserted += 1
        else:
            msg += "SKIP no pmid"
        print(msg)
    print(f"Inserted {num_inserted} new records")


@app.command()
def load_imap(db_path: str):
    """Load articles from an IMAP mailbox into the database."""

    load_dotenv()

    emails = fetch_emails(
        os.environ["IMAP_SERVER"],
        os.environ["IMAP_PORT"],
        os.environ["IMAP_USER"],
        os.environ["IMAP_PASS"],
    )

    records = []
    for envelope, message in emails:
        records.append(
            {
                "subject": envelope.subject.decode(),
                "from": f"{envelope.from_[0].mailbox.decode()}@{envelope.from_[0].host.decode()}",
                "sender": f"{envelope.sender[0].mailbox.decode()}@{envelope.sender[0].host.decode()}",
                "to": f"{envelope.to[0].mailbox.decode()}@{envelope.to[0].host.decode()}",
            }
        )
    df = pd.DataFrame(records)


@app.command()
def embed(db_path: str, model: str = "text-embedding-3-small"):
    """Embed articles in the database using the specified model."""
    load_dotenv()
    client = OpenAI()

    db = sqlite_utils.Database(db_path, tracer=tracer)
    table = db.table("articles")
    table.add_column(model, "BLOB")
    for article in track(table.rows_where("`text-embedding-3-small` is null")):
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
    logger.add(sys.stderr, level="INFO")
    # logger.add(sys.stderr, level="DEBUG")

    app()
