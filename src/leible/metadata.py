import requests
from bs4 import BeautifulSoup
from loguru import logger
from ratelimit import limits, sleep_and_retry

from leible.models import Article


def extract_article_from_pubmed_xml(soup: BeautifulSoup) -> Article:
    """Extract article metadata from PubMed XML.

    This is a helper function for parsing the XML returned by the NCBI Entrez
    API.

    Parameters
    ----------
    soup : BeautifulSoup
        BeautifulSoup object containing the PubMed XML for a single article.

    Returns
    -------
    Article
        Article object containing metadata for the article.
    """
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
    """Get articles by PMIDs from NCBI Entrez.

    Parameters
    ----------
    pmids : list[str]
        List of PubMed IDs (PMIDs) to fetch articles for.

    Returns
    -------
    list[Article]
        List of Article objects containing metadata for the requested PMIDs.
    """
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



@sleep_and_retry
@limits(calls=1, period=1)
def get_article_from_biorxiv(url: str) -> Article:
    """Get article metadata from BioRxiv.

    Parameters
    ----------
    url : str
        URL of the BioRxiv article.

    Returns
    -------
    Article
        Article object containing metadata for the article.
    """
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
