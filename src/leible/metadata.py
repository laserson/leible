import re
import requests
from bs4 import BeautifulSoup
from loguru import logger
from ratelimit import limits, sleep_and_retry
from urllib.parse import quote as urlquote
from semanticscholar import SemanticScholar
from toolz import partition_all

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
def request_articles_from_ncbi_entrez(pmids: list[str]) -> list[Article]:
    """Get articles by PMIDs from NCBI Entrez.

    Note: makes a single batch request for all PMIDs.

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
@limits(calls=1, period=3)
def request_articles_from_arxiv(ids: list[str]) -> list[Article]:
    """Get articles by ArXiv IDs.

    Note: makes a single batch request for all IDs.

    Parameters
    ----------
    ids : list[str]
        List of ArXiv IDs to fetch articles for.

    Returns
    -------
    list[Article]
        List of Article objects containing metadata for the requested ArXiv IDs.
    """
    logger.debug("Requesting {} articles from ArXiv", len(ids))
    max_results = 1000
    if len(ids) > max_results:
        raise ValueError(f"Max {max_results} IDs for one request")
    response = requests.post(
        "https://export.arxiv.org/api/query",
        data={"id_list": ",".join(ids), "start": 0, "max_results": max_results},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "xml")
    entries = soup.find_all("entry")
    articles = []
    for entry in entries:
        title = entry.find("title").get_text()
        abstract = entry.find("summary").get_text()
        journal = "arXiv"
        year = entry.find("published").get_text()[:4]
        authors = ", ".join(
            [author.find("name").get_text() for author in entry.find_all("author")]
        )
        arxiv = re.match(
            r"http://arxiv\.org/abs/([a-zA-Z0-9]+\.[a-zA-Z0-9]+)",
            entry.find("id").get_text(),
        ).group(1)
        article = Article(
            title=title,
            abstract=abstract,
            journal=journal,
            year=year,
            authors=authors,
            arxiv=arxiv,
        )
        articles.append(article)
    logger.debug("Returning {} articles from ArXiv", len(articles))
    return articles


@sleep_and_retry
@limits(calls=1, period=1)
def request_article_from_biorxiv_url(url: str) -> Article:
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


def extract_article_from_crossref_work_item(item: dict) -> Article:
    """Extract article metadata from a Crossref work item.

    Parameters
    ----------
    item : dict
        Crossref work item (returned from API).

    Returns
    -------
    Article
        Article object containing metadata for the article.
    """
    try:
        title = item["title"][0]
    except KeyError:
        raise ValueError("No title found for this article")

    try:
        abstract = item["abstract"]
    except KeyError:
        raise ValueError("No abstract found for this article")

    journal = item.get("container-title", [None])[0]

    if "published-print" in item:
        year = item["published-print"]["date-parts"][0][0]
    elif "published-online" in item:
        year = item["published-online"]["date-parts"][0][0]
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


def request_articles_from_crossref(
    dois: list[str], contact_email: str = None
) -> list[Article]:
    """Get articles from Crossref.

    Note: makes multiple API calls in batches of 20. Ratelimits to 1 call per
    second

    Parameters
    ----------
    dois : list[str]
        List of DOIs to fetch articles for.
    email : str, optional
        Email to use for the Polite pool for Crossref API requests.

    Returns
    -------
    list[Article]
        List of Article objects containing metadata for the requested DOIs.
    """
    logger.debug("Requesting {} articles in total from Crossref", len(dois))

    base_url = "https://api.crossref.org/works"
    headers = {}
    if contact_email:
        headers["User-Agent"] = f"Leible/0.1 (mailto:{contact_email})"

    @sleep_and_retry
    @limits(calls=1, period=1)
    def do_api_call(doi_batch: list[str]) -> list[dict]:
        doi_filter = ",".join(f"doi:{urlquote(doi)}" for doi in doi_batch)
        next_cursor = urlquote("*")
        url = f"{base_url}?filter={doi_filter}&cursor={next_cursor}"
        if len(url) > 1700:
            # this number was chosen as 2000 - len(cursor)
            raise ValueError(
                f"Too many dois for one url-encoded request ({len(url)} chars): {url}"
            )
        logger.debug("Fetching from Crossref: {}", url)
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        items = data.get("message", {}).get("items", [])
        logger.debug("Fetched {} items from Crossref", len(items))
        return items

    all_items = []
    for batch in partition_all(20, dois):
        items = do_api_call(batch)
        all_items.extend(items)

    logger.debug("Fetched {} items in total from Crossref", len(all_items))
    missing_dois = list(set(dois) - set([item["DOI"] for item in all_items]))
    logger.info(
        f"Failed to get {len(missing_dois)} of {len(dois)} DOIs from CrossRef: {missing_dois}"
    )

    articles = []
    for item in all_items:
        try:
            article = extract_article_from_crossref_work_item(item)
        except ValueError as e:
            logger.debug("{}: {}: {}", e, item["DOI"], item)
            continue
        logger.debug("Successfully extracted article: {}", article.doi)
        articles.append(article)

    logger.debug(
        "Successfully returned {} of {} articles from Crossref",
        len(articles),
        len(dois),
    )
    return articles


def request_articles_from_semantic_scholar(ids: list[str]) -> list[Article]:
    """Get articles by IDs from Semantic Scholar.

    Note: makes multiple API calls through SemanticScholar package with 500 IDs
    per call

    Parameters
    ----------
    ids : list[str]
        List of identifiers to search for. Usually DOIs, but can be any
        identifier type that Semantic Scholar recognizes.

    Returns
    -------
    list[Article]
        List of Article objects containing the matched papers' metadata.
    """
    logger.debug("Requesting {} papers from Semantic Scholar", len(ids))
    sch = SemanticScholar()
    papers = []
    missing_ids = []
    for batch in partition_all(500, ids):
        found, missing = sch.get_papers(batch, return_not_found=True)  # network call
        papers.extend(found)
        missing_ids.extend(missing)
    logger.info(f"Matched {len(papers)} papers, missing {len(missing_ids)}")
    for _id in missing_ids:
        logger.debug(f"Missing ID: {_id}")
    articles = []
    for paper in papers:
        if paper.title is None:
            logger.debug("Missing title for paper: {}", paper.externalIds["DOI"])
            continue
        if paper.abstract is None:
            logger.debug("Missing abstract for paper: {}", paper.externalIds["DOI"])
            continue
        articles.append(
            Article(
                title=paper.title,
                abstract=paper.abstract,
                year=paper.year,
                authors=", ".join([author.name for author in paper.authors]),
                doi=paper.externalIds.get("DOI"),
                pmid=paper.externalIds.get("PubMed"),
            )
        )
    logger.debug("Returning {} articles from Semantic Scholar", len(articles))
    return articles


def request_articles_by_dois(
    dois: list[str], contact_email: str = None
) -> list[Article]:
    """Get articles by DOIs.

    This is a wrapper that tries a few methods to get articles given a list of
    DOIs. The order of the returned articles may not match the order of the input
    DOIs.

    Parameters
    ----------
    dois : list[str]
        List of DOIs to fetch articles for.
    contact_email : str, optional
        Email to use for the Polite pool for Crossref API requests.

    Returns
    -------
    list[Article]
        List of Article objects containing metadata for the requested DOIs.
    """
    logger.info("Requesting {} articles by DOIs", len(dois))
    articles = []
    crossref_articles = request_articles_from_crossref(
        dois, contact_email=contact_email
    )
    logger.info("Found {} articles from Crossref", len(crossref_articles))
    articles.extend(crossref_articles)
    missing_dois = set(dois) - set(a.doi for a in crossref_articles)
    sch_articles = request_articles_from_semantic_scholar(missing_dois)
    logger.info("Found {} articles from Semantic Scholar", len(sch_articles))
    articles.extend(sch_articles)
    return articles
