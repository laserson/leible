import polars as pl
import requests
from ratelimit import limits, sleep_and_retry

from leible.metadata import request_articles_by_dois
from leible.models import Article


def load_readcube_papers_csv(csv_path: str, contact_email: str) -> list[Article]:
    df = pl.read_csv(csv_path, infer_schema=False).filter(
        pl.col("created (Read-Only)") > "2022-01-01"
    )
    dois = df.drop_nulls(subset="doi").get_column("doi").to_list()
    articles = request_articles_by_dois(dois, contact_email=contact_email)
    return articles


@sleep_and_retry
@limits(calls=5, period=1)
def doi_to_redirected_url(doi: str) -> str:
    response = requests.head(f"https://doi.org/{doi}", allow_redirects=True)
    return response.url
