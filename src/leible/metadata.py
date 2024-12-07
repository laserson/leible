import requests
from bs4 import BeautifulSoup
from loguru import logger
from ratelimit import limits, sleep_and_retry
from urllib.parse import quote as urlquote

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
        authors = []
        for author_soup in soup.find_all("Author"):
            last_name = author_soup.find("LastName").text.strip()
            first_name = author_soup.find("ForeName").text.strip()
            authors.append(f"{first_name} {last_name}")
        authors = ", ".join(authors)
    except AttributeError:
        authors = None

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
        title=title,
        abstract=abstract,
        journal=journal,
        year=year,
        authors=authors,
        doi=doi,
        pmid=pmid,
    )


@sleep_and_retry
@limits(calls=1, period=1)
def get_articles_from_ncbi_entrez(pmids: list[str]) -> list[Article]:
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
    authors_soup = soup.find("div", class_="highwire-cite-authors")
    authors = []
    for author_soup in authors_soup.find_all("span", class_="highwire-citation-author"):
        given_name = author_soup.find("span", class_="nlm-given-names").get_text()
        last_name = author_soup.find("span", class_="nlm-surname").get_text()
        authors.append(f"{given_name} {last_name}")
    authors = ", ".join(authors)
    doi = soup.find("meta", attrs={"name": "citation_doi"}).get("content")
    return Article(
        title=title,
        abstract=abstract,
        journal=journal,
        year=year,
        authors=authors,
        doi=doi,
    )


def extract_article_from_crossref_work_item(item):
    try:
        title = item["title"][0]
    except KeyError:
        raise ValueError("No title found for this article")

    try:
        abstract = item["abstract"]
    except KeyError:
        raise ValueError("No abstract found for this article")

    journal = item.get("container-title")

    if "published-print" in item:
        year = item["published-print"]["date-parts"][0]
    elif "published-online" in item:
        year = item["published-online"]["date-parts"][0]
    else:
        year = None

    authors = []
    for author in item.get("author", []):
        first_name = author.get("given", "")
        last_name = author.get("family", "")
        authors.append(f"{first_name} {last_name}")
    authors = ", ".join(authors)

    doi = item["DOI"]

    return Article(
        title=title,
        abstract=abstract,
        journal=journal,
        year=year,
        authors=authors,
        doi=doi,
    )


@sleep_and_retry
@limits(calls=1, period=1)
def get_articles_from_crossref(dois: list[str], email: str = None) -> list[Article]:
    """Get articles from Crossref."""
    logger.debug("Requesting {} articles in total from Crossref", len(dois))

    base_url = "https://api.crossref.org/works"
    headers = {}
    if email:
        headers["User-Agent"] = f"Leible/0.1 (mailto:{email})"
    doi_filter = ",".join(f"doi:{urlquote(doi)}" for doi in dois)
    all_items = []

    # first request
    next_cursor = urlquote("*")
    url = f"{base_url}?filter={doi_filter}&cursor={next_cursor}"
    if len(url) > 1700:
        # this number was chosen as 2000 - len(cursor)
        raise ValueError(f"Too many dois for one url-encoded request ({len(url)} chars): {url}")
    logger.debug("Fetching from Crossref: {}", url)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    items = data.get("message", {}).get("items", [])
    logger.debug("Fetched {} items from Crossref", len(items))
    all_items.extend(items)

    # if there's more, keep fetching
    while len(items) > 0:
        next_cursor = urlquote(data["message"]["next-cursor"])
        url = f"{base_url}?filter={doi_filter}&cursor={next_cursor}"
        logger.debug("Fetching more from Crossref: {}", url)
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        items = data.get("message", {}).get("items", [])
        logger.debug("Fetched additional {} items from Crossref", len(items))
        all_items.extend(items)

    logger.debug("Fetched {} items in total from Crossref", len(all_items))
    missing_dois = list(set(dois) - set([item["DOI"] for item in all_items]))
    logger.info(f"Failed to get {len(missing_dois)} of {len(dois)} DOIs from CrossRef: {missing_dois}")

    articles = []
    for item in all_items:
        try:
            article = extract_article_from_crossref_work_item(item)
        except ValueError as e:
            logger.debug("{}: {}: {}", e, item["DOI"], item)
            continue
        logger.debug("Successfully extracted article: {}", article.doi)
        articles.append(article)
    return articles
