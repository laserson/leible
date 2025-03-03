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
from tqdm import tqdm

from leible.models import Article
from leible.utils import (
    doi_to_redirected_url,
    get_first_simple_type,
    get_page_content_with_playwright,
    partition_by_predicate,
)


def get_text(
    soup: BeautifulSoup, selector: str, attrs: dict = None, separator: str = ""
) -> str | None:
    """Basically runs soup.find(selector).get_text() but gracefully handles None"""
    found = soup.find(selector, attrs=attrs) if soup else None
    return found.get_text().strip() if found else None


def unwrap(text: str) -> str:
    return (
        " ".join(chunk.strip() for chunk in text.strip().split("\n")) if text else None
    )


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
    match = (
        re.match(r"https?://arxiv\.org/abs/([0-9]+\.[0-9]+)(v[0-9]+)?.*", publisher_url)
        if publisher_url
        else None
    )
    arxiv = match.group(1) if match else None
    doi = f"10.48550/arXiv.{arxiv}"
    return Article(
        arxiv=arxiv,
        doi=doi,
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
    if len(ids) == 0:
        return []
    logger.info("Requesting {} IDs from ArXiv", len(ids))
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
    num_missing = len(ids) - len(articles)
    if num_missing > 0:
        logger.info("Missed {} IDs from ArXiv: {}", num_missing, ids)
    return articles


def extract_article_from_openreview_html(soup: BeautifulSoup) -> Article:
    """Extract article metadata from OpenReview HTML."""
    authors = ", ".join(
        [
            author_soup.get("content")
            for author_soup in soup.find_all("meta", attrs={"name": "citation_author"})
        ]
    )
    pdf_url = soup.find("meta", attrs={"name": "citation_pdf_url"}).get("content")
    publisher_url = re.sub(r"openreview\.net/pdf\?", "openreview.net/forum?", pdf_url)
    return Article(
        publisher_url=publisher_url,
        title=soup.find("meta", attrs={"name": "citation_title"}).get("content"),
        abstract=soup.find("meta", attrs={"name": "citation_abstract"}).get("content"),
        authors=authors,
        year=int(
            soup.find("meta", attrs={"name": "citation_online_date"}).get("content")[:4]
        ),
        journal=soup.find("meta", attrs={"name": "citation_conference_title"}).get(
            "content"
        ),
    )


@sleep_and_retry
@limits(calls=1, period=2)
def request_article_from_openreview_url(url: str) -> Article:
    """Get article metadata from OpenReview."""
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    article = extract_article_from_openreview_html(soup)
    return article


@sleep_and_retry
@limits(calls=1, period=2)
def request_article_from_elsevier_pii(pii: str, api_key: str) -> Article:
    """Get article metadata from Elsevier API."""
    url = f"https://api.elsevier.com/content/article/pii/{pii}"
    response = requests.get(
        url,
        params={"view": "META_ABS"},
        headers={"Accept": "application/json", "X-ELS-APIKey": api_key},
    )
    response.raise_for_status()
    data = response.json()["full-text-retrieval-response"]
    return Article(
        doi=data["coredata"]["prism:doi"],
        pmid=data["pubmed-id"],
        publisher_url=next(
            (
                link["@href"]
                for link in data["coredata"]["link"]
                if link["@rel"] == "scidir"
            ),
            None,
        ),
        title=data["coredata"]["dc:title"],
        abstract=unwrap(data["coredata"]["dc:description"]),
        journal=data["coredata"]["prism:publicationName"],
        year=int(data["coredata"]["prism:coverDate"][:4]),
        authors=", ".join([author["$"] for author in data["coredata"]["dc:creator"]]),
    )


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
@limits(calls=1, period=2)
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
    # BioRxiv sometimes(?) does something weird and redirects from https to
    # http, even if you start with https. But BioRxiv returns a null page if you
    # make a request to an http URL. So we manually handle the redirects to swap
    # out http for https if necessary.
    https_url = re.sub(r"^http://", "https://", url)
    response = requests.get(https_url, allow_redirects=False)
    response.raise_for_status()
    while 300 <= response.status_code < 400:
        url = response.headers["Location"]
        https_url = re.sub(r"^http://", "https://", url)
        response = requests.get(https_url, allow_redirects=False)
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    article = extract_article_from_biorxiv_html(soup)
    return article


def extract_article_from_nature_html(soup: BeautifulSoup) -> Article:
    """Extract article metadata from Nature HTML."""
    authors = [
        author_soup.get("content")
        for author_soup in soup.find_all(
            "meta", attrs={"name": ignore_case_re("dc.creator")}
        )
    ]
    last_first_pairs = [author.split(",") for author in authors]
    authors = ", ".join(
        [f"{first.strip()} {last.strip()}" for last, first in last_first_pairs]
    )
    return Article(
        doi=soup.find("meta", attrs={"name": ignore_case_re("citation_doi")}).get(
            "content"
        ),
        publisher_url=soup.find("link", rel=ignore_case_re("canonical")).get("href"),
        title=soup.find("meta", attrs={"name": ignore_case_re("dc.title")}).get(
            "content"
        ),
        abstract=soup.find(
            "meta", attrs={"name": ignore_case_re("dc.description")}
        ).get("content"),
        journal=soup.find(
            "meta", attrs={"name": ignore_case_re("citation_journal_title")}
        ).get("content"),
        year=int(
            soup.find("meta", attrs={"name": ignore_case_re("dc.date")}).get("content")[
                :4
            ]
        ),
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
                for author_soup in soup.find_all(
                    "meta", attrs={"name": ignore_case_re("dc.creator")}
                )
            ]
        )
        article = Article(
            doi=soup.find(
                "meta", attrs={"name": ignore_case_re("publication_doi")}
            ).get("content"),
            publisher_url=soup.find("link", rel=ignore_case_re("canonical")).get(
                "href"
            ),
            title=soup.find("meta", attrs={"name": ignore_case_re("dc.title")}).get(
                "content"
            ),
            abstract=soup.find(
                "meta", attrs={"name": ignore_case_re("dc.description")}
            ).get("content"),
            journal=soup.find(
                "meta", attrs={"name": ignore_case_re("citation_journal_title")}
            ).get("content"),
            year=int(
                soup.find("meta", attrs={"name": ignore_case_re("dc.date")}).get(
                    "content"
                )[:4]
            ),
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
    logger.info("Requesting {} DOIs from Semantic Scholar", len(dois))
    sch = SemanticScholar(api_key=api_key, timeout=60)

    @sleep_and_retry
    @limits(calls=1, period=4)
    def do_api_call(doi_batch: list[str]) -> tuple[list[Paper], list[str]]:
        found, missing = sch.get_papers(
            doi_batch, return_not_found=True
        )  # network call
        return found, missing

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


def simple_load_readcube_papers_csv(csv_path: str) -> list[Article]:
    df = pl.read_csv(csv_path, infer_schema=False)
    return [
        Article(title=row["title"], abstract=row["abstract"])
        for row in df.iter_rows(named=True)
    ]


def load_readcube_papers_csv(
    csv_path: str, contact_email: str, s2_api_key: str = None, elsevier_api_key=None
) -> list[Article]:
    """Load ReadCube-exported CSV and return list of articles.

    This function is disgusting because it deals with a lot of gross issues in
    the ReadCube dataset.

    Parameters
    ----------
    csv_path : str
        Path to the ReadCube-exported CSV file.
    contact_email : str
        Email to use for the Polite pool for Crossref API requests.
    s2_api_key : str, optional
        API key for Semantic Scholar.

    Returns
    -------
    list[Article]
        List of Article objects containing the matched papers' metadata.
    """
    logger.info("Loading ReadCube-exported CSV {}", csv_path)
    df = pl.read_csv(csv_path, infer_schema=False)
    logger.info("Total {} rows in ReadCube CSV", len(df))

    articles = []

    # DOI-based loading
    dois = df.drop_nulls(subset="doi").get_column("doi").to_list()
    logger.info("Found {} DOIs in ReadCube CSV; attempting to load...", len(dois))

    # try Semantic Scholar for DOIs
    counter = 0
    try:
        requested_articles = request_articles_from_semantic_scholar(
            dois, api_key=s2_api_key
        )
    except Exception as e:
        logger.warning("Semantic Scholar request failed: {}", e)
        requested_articles = []
    for article in requested_articles:
        if article.title is not None and article.abstract is not None:
            articles.append(article)
            counter += 1
    logger.info(f"Successfully loaded {counter} articles from Semantic Scholar by DOI")
    missing_dois = list(set(dois) - set([article.doi for article in articles]))

    # try Crossref for DOIs
    counter = 0
    try:
        requested_articles = request_articles_from_crossref(
            missing_dois, contact_email=contact_email
        )
    except Exception as e:
        logger.warning("Crossref request failed: {}", e)
        requested_articles = []
    for article in requested_articles:
        if article.title is not None and article.abstract is not None:
            articles.append(article)
            counter += 1
    logger.info(f"Successfully loaded {counter} articles from Crossref by DOI")
    missing_dois = list(set(dois) - set([article.doi for article in articles]))

    # try Nature DOIs directly
    nature_dois = list(filter(lambda doi: doi.startswith("10.1038/"), missing_dois))
    counter = 0
    for doi in tqdm(nature_dois, desc="Getting Nature articles directly"):
        nature_id = doi.split("/")[-1]
        url = f"https://www.nature.com/articles/{nature_id}"
        try:
            articles.append(request_article_from_nature_url(url))
            counter += 1
        except Exception:
            logger.debug("Failed to get article metadata from Nature: {}", doi)
            continue
    logger.info(f"Successfully loaded {counter} articles from Nature by DOI")
    missing_dois = list(set(dois) - set([article.doi for article in articles]))

    # try arXiv DOIs directly
    arxiv_dois = list(
        filter(lambda doi: doi.startswith("10.48550/arxiv"), missing_dois)
    )
    arxiv_pattern = re.compile(r"10.48550/arxiv\.([0-9]+\.[0-9]+)(v[0-9]+)?")
    arxiv_ids = [
        match.group(1) for doi in arxiv_dois if (match := arxiv_pattern.match(doi))
    ]
    try:
        arxiv_articles = request_articles_from_arxiv(arxiv_ids)
        counter += len(arxiv_articles)
    except Exception as e:
        logger.warning("arXiv request failed: {}\n{}", e, arxiv_dois)
        arxiv_articles = []
    articles.extend(arxiv_articles)
    logger.info(
        f"Successfully loaded {len(arxiv_articles)} of {len(arxiv_dois)} articles from arXiv by DOI"
    )
    missing_dois = list(set(dois) - set([article.doi for article in articles]))

    # try Elsevier DOIs directly (through ScienceDirect)
    elsevier_dois = list(filter(lambda doi: doi.startswith("10.1016/"), missing_dois))
    counter = 0
    for doi in tqdm(elsevier_dois, desc="Getting Elsevier articles directly"):
        try:
            elsevier_url_1 = doi_to_redirected_url(doi)
            elsevier_id = elsevier_url_1.split("/")[-1]
            articles.append(
                request_article_from_elsevier_pii(elsevier_id, elsevier_api_key)
            )
            counter += 1
        except Exception as e:
            logger.debug("Failed to get article metadata from Elsevier: {}\n{}", doi, e)
            continue
    logger.info(
        f"Successfully loaded {counter} of {len(elsevier_dois)} articles from Elsevier by DOI"
    )
    missing_dois = list(set(dois) - set([article.doi for article in articles]))
    logger.info(
        "Missed {} DOIs from ReadCube CSV:\n{}", len(missing_dois), sorted(missing_dois)
    )

    # Non-DOI loading
    counter = len(df.filter(pl.col("doi").is_null()))
    logger.info(f"Fetching {counter} articles without DOIs...")

    # try arXiv IDs
    arxiv_ids = (
        df.filter(pl.col("doi").is_null())
        .drop_nulls(subset="arxiv")
        .get_column("arxiv")
        .to_list()
    )
    counter = 0
    try:
        arxiv_articles = request_articles_from_arxiv(arxiv_ids)
        counter += len(arxiv_articles)
    except Exception as e:
        logger.warning("arXiv request failed: {}", e)
        arxiv_articles = []
    articles.extend(arxiv_articles)
    logger.info(
        f"Successfully loaded {counter} of {len(arxiv_ids)} articles with arXiv IDs"
    )

    # try URLs in ReadCube CSV
    urls = (
        df.filter(pl.col("doi").is_null())
        .drop_nulls(subset="url")
        .get_column("url")
        .to_list()
    )

    # try arXiv URLs
    arxiv_urls, other_urls = partition_by_predicate(
        lambda url: url.startswith("https://arxiv.org/abs/"), urls
    )
    arxiv_pattern = re.compile(r"https?://arxiv\.org/abs/([0-9]+\.[0-9]+)(v[0-9]+)?.*")
    arxiv_ids = [
        match.group(1) for url in arxiv_urls if (match := arxiv_pattern.match(url))
    ]
    try:
        arxiv_articles = request_articles_from_arxiv(arxiv_ids)
    except Exception as e:
        logger.warning("arXiv request failed: {}", e)
        arxiv_articles = []
    articles.extend(arxiv_articles)
    logger.info(
        f"Successfully loaded {len(arxiv_articles)} of {len(arxiv_urls)} articles with arXiv URLs"
    )

    # try OpenReview URLs
    openreview_urls, other_urls = partition_by_predicate(
        lambda url: url.startswith("https://openreview.net/forum?id="), other_urls
    )
    counter = 0
    for url in tqdm(openreview_urls, desc="Getting OpenReview articles"):
        try:
            articles.append(request_article_from_openreview_url(url))
            counter += 1
        except Exception:
            logger.debug("Failed to get article metadata from OpenReview: {}", url)
            continue
    logger.info(
        f"Successfully loaded {counter} of {len(openreview_urls)} articles from OpenReview"
    )
    logger.info("Leftover URLs: {}", other_urls)
    logger.info("Total articles loaded: {} of {}", len(articles), len(df))

    return articles
