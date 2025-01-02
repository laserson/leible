import re
from urllib.parse import quote as urlquote

import polars as pl
import requests
from bs4 import BeautifulSoup
from loguru import logger
from ratelimit import limits, sleep_and_retry
from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper
from toolz import partition_all
from tqdm.auto import tqdm

from leible.models import Article
from leible.utils import get_first_simple_type, get_page_content_with_playwright


def get_text(
    soup: BeautifulSoup, selector: str, attrs: dict = None, separator: str = ""
) -> str | None:
    """Basically runs soup.find(selector).get_text() but gracefully handles None"""
    found = soup.find(selector, attrs=attrs) if soup else None
    return found.get_text().strip() if found else None


def unwrap(text: str) -> str:
    return " ".join(chunk.strip() for chunk in text.split("\n"))


def ignore_case_re(content: str) -> re.Pattern:
    content = re.escape(content)
    return re.compile(f"^{content}$", re.IGNORECASE)


def extract_article_from_pubmed_xml(soup: BeautifulSoup) -> Article:
    """Extract article metadata from PubMed XML."""
    authors = [
        (
            get_text(author_soup, "ForeName") or "",
            get_text(author_soup, "LastName") or "",
        )
        for author_soup in soup.find_all("Author")
    ]
    authors = (
        ", ".join([f"{first} {last}" for first, last in authors])
        if len(authors) > 0
        else None
    )
    id_list = soup.find("ArticleIdList")
    return Article(
        doi=get_text(id_list, "ArticleId", attrs={"IdType": "doi"}),
        pmid=get_text(id_list, "ArticleId", attrs={"IdType": "pubmed"}),
        title=unwrap(get_text(soup, "ArticleTitle")),
        abstract=unwrap(get_text(soup, "AbstractText")),
        journal=get_text(soup.find("Journal"), "Title"),
        year=int(get_text(soup.find("PubDate"), "Year")),
        authors=authors,
    )


@sleep_and_retry
@limits(calls=1, period=1)
def request_articles_from_ncbi_entrez(
    pmids: list[str], contact_email: str
) -> list[Article]:
    """Get articles by PMIDs from NCBI Entrez.

    Note: makes a single batch request for all PMIDs.

    Parameters
    ----------
    pmids : list[str]
        List of PubMed IDs (PMIDs) to fetch articles for.
    contact_email : str
        Email to use to identify requests to NCBI Entrez API.

    Returns
    -------
    list[Article]
        List of Article objects containing metadata for the requested PMIDs.
    """
    logger.debug("Requesting {} PMIDs from NCBI Entrez", len(pmids))
    response = requests.post(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        data={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": "leible",
            "email": contact_email,
        },
    )
    response.raise_for_status()
    articles = [
        extract_article_from_pubmed_xml(article_soup)
        for article_soup in BeautifulSoup(response.text, "xml").find_all(
            "PubmedArticle"
        )
    ]
    missing_pmids = list(set(pmids) - set([article.pmid for article in articles]))
    if len(missing_pmids) > 0:
        logger.debug(
            "Missed {} PMIDs from NCBI Entrez: {}", len(missing_pmids), missing_pmids
        )
    return articles


def extract_article_from_arxiv_xml(soup: BeautifulSoup) -> Article:
    """Extract article metadata from ArXiv XML."""
    authors = [get_text(author, "name") for author in soup.find_all("author")]
    authors = ", ".join(authors) if len(authors) > 0 else None
    publisher_url = get_text(soup, "id")
    arxiv = (
        re.match(
            r"http://arxiv\.org/abs/([a-zA-Z0-9]+\.[a-zA-Z0-9]+)", publisher_url
        ).group(1)
        if publisher_url
        else None
    )
    return Article(
        arxiv=arxiv,
        title=unwrap(get_text(soup, "title")),
        abstract=unwrap(get_text(soup, "summary")),
        journal="arXiv",
        year=(date := get_text(soup, "published")) and int(date[:4]),
        authors=authors,
    )


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
    logger.debug("Requesting {} IDs from ArXiv", len(ids))
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
    articles = [extract_article_from_arxiv_xml(entry_soup) for entry_soup in entries]
    missing_ids = list(set(ids) - set([article.arxiv for article in articles]))
    if len(missing_ids) > 0:
        logger.debug("Missed {} IDs from ArXiv: {}", len(missing_ids), missing_ids)
    return articles


def extract_article_from_biorxiv_html(soup: BeautifulSoup) -> Article:
    """Extract article metadata from BioRxiv HTML."""
    publisher_url = soup.find("meta", attrs={"name": "citation_public_url"}).get(
        "content"
    )
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
    return Article(
        doi=soup.find("meta", attrs={"name": "citation_doi"}).get("content"),
        publisher_url=publisher_url,
        title=get_text(soup, "h1", attrs={"id": "page-title"}),
        abstract=get_text(soup, "div", attrs={"class": "abstract"}, separator="\n"),
        journal="BioRxiv",
        year=year,
        authors=authors,
    )


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
    article = extract_article_from_biorxiv_html(soup)
    return article


def extract_article_from_nature_html(soup: BeautifulSoup) -> Article:
    """Extract article metadata from Nature HTML."""
    authors = [
        author_soup.get("content")
        for author_soup in soup.find_all("meta", name=ignore_case_re("dc.creator"))
    ]
    last_first_pairs = [author.split(",") for author in authors]
    authors = ", ".join(
        [f"{first.strip()} {last.strip()}" for last, first in last_first_pairs]
    )
    return Article(
        doi=soup.find("meta", name=ignore_case_re("citation_doi")).get("content"),
        publisher_url=soup.find("link", rel=ignore_case_re("canonical")).get("href"),
        title=soup.find("meta", name=ignore_case_re("dc.title")).get("content"),
        abstract=soup.find("meta", name=ignore_case_re("dc.description")).get(
            "content"
        ),
        journal=soup.find("meta", name=ignore_case_re("citation_journal_title")).get(
            "content"
        ),
        year=int(soup.find("meta", name=ignore_case_re("dc.date")).get("content")[:4]),
        authors=authors,
    )


@sleep_and_retry
@limits(calls=1, period=1)
def request_article_from_nature_url(url: str) -> Article:
    """Get article metadata from Nature.

    Parameters
    ----------
    url : str
        URL of the Nature article.

    Returns
    -------
    Article
        Article object containing metadata for the article.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    article = extract_article_from_nature_html(soup)
    return article


def extract_article_from_science_html(soup: BeautifulSoup) -> Article:
    """Extract article metadata from Science HTML."""
    try:
        authors = ", ".join(
            [
                author_soup.get("content")
                for author_soup in soup.find_all("meta", name=ignore_case_re("dc.creator"))
            ]
        )
        article = Article(
            doi=soup.find("meta", name=ignore_case_re("publication_doi")).get("content"),
            publisher_url=soup.find("link", rel=ignore_case_re("canonical")).get("href"),
            title=soup.find("meta", name=ignore_case_re("dc.title")).get("content"),
            abstract=soup.find("meta", name=ignore_case_re("dc.description")).get(
                "content"
            ),
            journal=soup.find("meta", name=ignore_case_re("citation_journal_title")).get(
                "content"
            ),
            year=int(soup.find("meta", name=ignore_case_re("dc.date")).get("content")[:4]),
            authors=authors,
        )
    except Exception as e:
        url = soup.find("link", rel=ignore_case_re("canonical"))
        url = url.get("href") if url else None
        logger.debug("Failed to extract article metadata from Science: {}", url)
        raise e
    return article


@sleep_and_retry
@limits(calls=1, period=1)
def request_article_from_science_url(url: str) -> Article:
    content = get_page_content_with_playwright(url)
    soup = BeautifulSoup(content, "html.parser")
    article = extract_article_from_science_html(soup)
    return article


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
    if "published-print" in item:
        year = get_first_simple_type(item["published-print"]["date-parts"])
    elif "published-online" in item:
        year = get_first_simple_type(item["published-online"]["date-parts"])
    else:
        year = None
    authors = []
    for author in item.get("author", []):
        first_name = author.get("given", "")
        last_name = author.get("family", "")
        authors.append(f"{first_name} {last_name}")
    authors = ", ".join(authors)
    abstract = get_first_simple_type(item.get("abstract", [None]))
    return Article(
        doi=item["DOI"],
        publisher_url=item.get("resource", {}).get("primary", {}).get("URL", None),
        title=get_first_simple_type(item.get("title", [None])),
        abstract=abstract,
        # abstract=get_first_simple_type(item.get("abstract", [None])),
        journal=get_first_simple_type(item.get("container-title", [None])),
        year=year,
        authors=authors,
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
    logger.debug("Requesting {} DOIs from Crossref", len(dois))

    base_url = "https://api.crossref.org/works"
    headers = {}
    if contact_email:
        headers["User-Agent"] = f"Leible/0.1 (mailto:{contact_email})"

    @sleep_and_retry
    @limits(calls=1, period=1)
    def do_api_call(doi_batch: list[str]) -> list[dict]:
        doi_filter = ",".join(f"doi:{urlquote(doi)}" for doi in doi_batch)
        url = f"{base_url}?filter={doi_filter}"
        if len(url) > 2000:
            raise ValueError(
                f"Too many dois for one url-encoded request ({len(url)} chars): {url}"
            )
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        items = data.get("message", {}).get("items", [])
        return items

    all_items = []
    for batch in partition_all(20, dois):
        items = do_api_call(batch)
        all_items.extend(items)
    missing_dois = list(set(dois) - set([item["DOI"] for item in all_items]))
    if len(missing_dois) > 0:
        logger.debug(
            "Missed {} DOIs from Crossref: {}", len(missing_dois), missing_dois
        )
    articles = [extract_article_from_crossref_work_item(item) for item in all_items]
    return articles


def sch_paper_to_article(paper: Paper) -> Article:
    """Convert a `semanticscholar.Paper` to `leible.models.Article`."""
    return Article(
        doi=paper.externalIds.get("DOI"),
        pmid=paper.externalIds.get("PubMed"),
        arxiv=paper.externalIds.get("ArXiv"),
        semantic_scholar=paper.paperId,
        title=paper.title,
        abstract=paper.abstract,
        journal=paper.journal.name if paper.journal else None,
        year=paper.year,
        authors=", ".join([author.name for author in paper.authors]),
    )


def request_articles_from_semantic_scholar(
    dois: list[str], api_key: str = None
) -> list[Article]:
    """Get articles by DOIs from Semantic Scholar.

    Note: makes multiple API calls through SemanticScholar package with 500 IDs
    per call

    Parameters
    ----------
    dois : list[str]
        List of DOIs to search for.

    Returns
    -------
    list[Article]
        List of Article objects containing the matched papers' metadata.
    """
    logger.debug("Requesting {} IDs from Semantic Scholar", len(dois))

    @sleep_and_retry
    @limits(calls=1, period=4)
    def do_api_call(doi_batch: list[str]) -> tuple[list[Paper], list[str]]:
        found, missing = sch.get_papers(
            doi_batch, return_not_found=True
        )  # network call
        return found, missing

    sch = SemanticScholar(api_key=api_key)
    papers = []
    missing_dois = []
    for batch in partition_all(500, dois):
        found, missing = do_api_call(batch)
        papers.extend(found)
        missing_dois.extend(missing)

    if len(missing_dois) > 0:
        logger.debug(
            "Missed {} DOIs from Semantic Scholar: {}", len(missing_dois), missing_dois
        )

    articles = [sch_paper_to_article(paper) for paper in papers]
    return articles


def request_articles_from_nature_dois(
    dois: list[str], s2_api_key: str = None
) -> list[Article]:
    articles = []
    requested_articles = request_articles_from_semantic_scholar(
        dois, api_key=s2_api_key
    )
    for article in requested_articles:
        if article.title is not None and article.abstract is not None:
            articles.append(article)
    missing_dois = set(dois) - set([article.doi for article in articles])
    # try crossref
    requested_articles = request_articles_from_crossref(missing_dois)
    for article in requested_articles:
        if article.title is not None and article.abstract is not None:
            articles.append(article)
    missing_dois = set(dois) - set([article.doi for article in articles])
    # try Nature pages directly
    for doi in tqdm(missing_dois):
        nature_id = doi.split("/")[-1]
        url = f"https://www.nature.com/articles/{nature_id}"
        try:
            articles.append(request_article_from_nature_url(url))
        except Exception:
            logger.debug("Failed to get article metadata from Nature DOI: {}", doi)
            continue
    return articles


def load_readcube_papers_csv(
    csv_path: str, contact_email: str, s2_api_key: str = None
) -> list[Article]:
    df = pl.read_csv(csv_path, infer_schema=False).filter(
        pl.col("created (Read-Only)") > "2022-01-01"
    )
    logger.info("Attempting to load {} papers from ReadCube", len(df))
    dois = df.drop_nulls(subset="doi").get_column("doi").to_list()
    articles = []
    requested_articles = request_articles_from_semantic_scholar(
        dois, api_key=s2_api_key
    )
    for article in requested_articles:
        if article.title is not None and article.abstract is not None:
            articles.append(article)
    missing_dois = list(set(dois) - set([article.doi for article in articles]))
    requested_articles = request_articles_from_crossref(
        missing_dois, contact_email=contact_email
    )
    for article in requested_articles:
        if article.title is not None and article.abstract is not None:
            articles.append(article)
    missing_dois = list(set(dois) - set([article.doi for article in articles]))
    # try Nature pages directly
    nature_dois = list(filter(lambda doi: doi.startswith("10.1038/"), missing_dois))
    for doi in tqdm(nature_dois):
        nature_id = doi.split("/")[-1]
        url = f"https://www.nature.com/articles/{nature_id}"
        try:
            articles.append(request_article_from_nature_url(url))
        except Exception:
            logger.debug("Failed to get article metadata from Nature: {}", doi)
            continue
    logger.info("Successfully loaded {} papers from ReadCube", len(articles))
    return articles
