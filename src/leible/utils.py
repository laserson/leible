import polars as pl
import requests
from loguru import logger
from ratelimit import limits, sleep_and_retry
from semanticscholar import SemanticScholar
from toolz import partition_all

from leible.models import Article


def load_readcube_papers_csv(csv_path: str) -> list[Article]:
    df = pl.read_csv(csv_path, infer_schema=False).filter(
        (pl.col("created (Read-Only)") > "2022-01-01") | (pl.col("year") > "2019")
    )
    dois = df.drop_nulls(subset="doi").get_column("doi").to_list()
    sch = SemanticScholar()
    papers = []
    missing_ids = []
    for batch in partition_all(500, dois):
        found, missing = sch.get_papers(batch, return_not_found=True)  # network call
        papers.extend(found)
        missing_ids.extend(missing)
    logger.info(f"Found {len(papers)} papers, missing {len(missing_ids)}")
    for _id in missing_ids:
        logger.debug(f"Missing DOI: {_id}")
    return [Article(title=p.title, abstract=p.abstract) for p in papers]


@sleep_and_retry
@limits(calls=5, period=1)
def doi_to_redirected_url(doi: str) -> str:
    response = requests.head(f"https://doi.org/{doi}", allow_redirects=True)
    return response.url
