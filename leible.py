import numpy as np
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
def get_article_embedding(client: OpenAI, model: str, title, abstract) -> np.ndarray:
    input = f"{title}\n\n{abstract}"
    response = client.embeddings.create(input=input, model=model)
    return np.asarray(response.data[0].embedding)


app = typer.Typer()


@app.command()
def load(db_path: str, input_csv: str):
    """Load articles from a CSV into the database.

    Currently, must be a `pmid` column.

    The end result should be that each record in the database has a well-formed
    Title and Abstract.
    """
    db = sqlite_utils.Database(db_path)
    ensure_article_table(db)
    table = db.table("articles")

    input_df = pd.read_csv(input_csv, dtype=str)
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
        else:
            msg += "SKIP No PMID"
        print(msg)


@app.command()
def embed(db_path: str, model: str = "text-embedding-3-small"):
    """Embed articles in the database using the specified model."""
    load_dotenv()
    client = OpenAI()

    db = sqlite_utils.Database(db_path)
    table = db.table("articles")
    for article in track(table.rows_where("embedding is null")):
        embedding = get_article_embedding(
            client, model, article["title"], article["abstract"]
        )
        table.update(article["id"], {"embedding": embedding})


@app.command()
def cluster(database: str):
    pass


if __name__ == "__main__":
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # logger.add(sys.stderr, level="DEBUG")

    app()
