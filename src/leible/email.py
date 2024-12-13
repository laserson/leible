import base64
import email
import email.policy
import io
import re
from base64 import urlsafe_b64decode
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from bs4 import BeautifulSoup
from imapclient import IMAPClient
from loguru import logger

from leible.metadata import (
    request_article_from_biorxiv_url,
    request_article_from_nature_url,
    request_article_from_science_url,
    request_articles_from_arxiv,
    request_articles_from_crossref,
    request_articles_from_ncbi_entrez,
    request_articles_from_semantic_scholar,
)
from leible.models import Article


def fetch_emails(
    imap_host, imap_port, imap_user, imap_pass, folder="ToC"
) -> list[email.message.EmailMessage]:
    with IMAPClient(host=imap_host, port=imap_port, use_uid=True) as client:
        client.login(imap_user, imap_pass)
        client.select_folder(folder, readonly=True)
        message_ids = client.search()
        emails = []
        for message_uid, message_data in client.fetch(
            message_ids, [b"RFC822", b"FLAGS"]
        ).items():
            message = email.message_from_bytes(
                message_data[b"RFC822"], policy=email.policy.default
            )
            message.add_header("X-IMAP-UID", str(message_uid))
            message.add_header("X-IMAP-FLAGS", str(message_data[b"FLAGS"]))
            emails.append(message)
        return emails


def extract_html_from_email(message: email.message.EmailMessage) -> BeautifulSoup:
    return BeautifulSoup(
        message.get_body(("html", "plain")).get_payload(decode=True).decode("utf-8"),
        "html.parser",
    )


def extract_urls_from_biorxiv_email(message: email.message.EmailMessage) -> list[str]:
    """Returns list of BioRxiv URLs"""
    soup = extract_html_from_email(message)
    prefixes = [
        "http://www.biorxiv.org/cgi/content/abstract",
        "http://www.biorxiv.org/content/early",
        "http://biorxiv.org/cgi/content/abstract",
        "http://biorxiv.org/content/early",
    ]
    urls = []
    for elt in soup.find_all("a"):
        url = elt.get("href")
        if any(url.startswith(prefix) for prefix in prefixes):
            urls.append(url)
    return urls


def extract_ids_from_arxiv_email(message: email.message.EmailMessage) -> list[str]:
    """Returns list of ArXiv IDs"""
    soup = extract_html_from_email(message)
    ids = [
        match.group(1)
        for line in soup.get_text().split("\n")
        if (match := re.fullmatch(r"^arXiv:([a-zA-Z0-9]+\.[a-zA-Z0-9]+)", line.strip()))
    ]
    return ids


def extract_pmids_from_pubmed_email(message: email.message.EmailMessage) -> list[str]:
    """Returns list of PubMed IDs"""
    soup = extract_html_from_email(message)
    anchors = soup.find_all("a", class_="docsum-title")
    pmids = [re.fullmatch(r"article_id=(\d+)", a.get("ref")).group(1) for a in anchors]
    return pmids


def extract_dois_from_nature_alert_email(
    message: email.message.EmailMessage,
) -> list[str]:
    """Returns list of DOIs from a "Nature Alert" email"""
    npg_magic_param = "_L54AD1F204_"
    npg_doi_prefix = "10.1038"
    soup = extract_html_from_email(message)
    dois = []
    for elt in soup.find_all("a"):
        href = elt.get("href")
        if href is None:
            continue
        params = parse_qs(urlparse(href).query)
        encoded_data = params.get(npg_magic_param)
        if encoded_data is None:
            continue
        # add padding
        encoded_data = encoded_data[0] + "=" * (4 - len(encoded_data[0]) % 4)
        decoded_data = urlsafe_b64decode(encoded_data).decode("utf-8")
        decoded_params = parse_qs(decoded_data)
        target_url = decoded_params.get("target")
        if target_url is None:
            continue
        target_url = urlparse(target_url[0])
        if not target_url.path.startswith("/articles"):
            continue
        article_id = re.fullmatch(r"/articles/(.+)", target_url.path).group(1)
        doi = f"{npg_doi_prefix}/{article_id}"
        dois.append(doi)
    return dois


def extract_dois_from_nature_ealerts_email(
    message: email.message.EmailMessage,
) -> list[str]:
    """Returns list of DOIs from "Nature eAlerts" email"""
    soup = extract_html_from_email(message)
    dois = re.findall(r"\bdoi:([^\s]+)", soup.get_text())
    return dois


def extract_ids_from_semantic_scholar_email(
    message: email.message.EmailMessage,
) -> list[str]:
    """Returns list of Semantic Scholar internal IDs"""
    soup = extract_html_from_email(message)
    urls = [elt.get("href") for elt in soup.find_all("a", class_="paper-link")]
    ids = [urlparse(url).path.split("/")[-1] for url in urls]
    return ids


def extract_urls_from_science_email(message: email.message.EmailMessage) -> list[str]:
    logger.debug("Extracting URLs from Science email")
    soup = extract_html_from_email(message)
    urls = [
        elt.get("href")
        for elt in soup.find_all("a", string=re.compile(r"Read More", re.IGNORECASE))
    ]
    logger.debug("Extracted {} URLs from Science email", len(urls))
    return urls


def process_email(
    message: email.message.EmailMessage, contact_email: str = None
) -> list[Article]:
    """Process an email and return a list of articles

    This is a "top-level" function that dispatches to the correct handler for the
    email sender and encodes necessary logic for each sender.

    Parameters
    ----------
    message : email.message.EmailMessage
        The email message to process
    contact_email : str
        Your contact email for Crossref "Polite" requests

    Returns
    -------
    list[Article]
        List of articles extracted from the email
    """
    _, from_ = email.utils.parseaddr(message.get("From"))
    _, to = email.utils.parseaddr(message.get("To"))
    subject = message.get("Subject")
    logger.info("Processing email from {} to {}: {}", from_, to, subject)
    articles = []
    if from_ == "cshljnls-mailer@alerts.highwire.org":
        urls = list(set(extract_urls_from_biorxiv_email(message)))
        logger.info("Found {} URLs in BioRxiv email", len(urls))
        for url in urls:
            try:
                articles.append(request_article_from_biorxiv_url(url))
            except Exception:
                logger.info("Failed to get article metadata from BioRxiv: {}", url)
                continue
        logger.info(
            "Found {} articles from {} URLs in BioRxiv email", len(articles), len(urls)
        )
        return articles
    if from_ == "efback@ncbi.nlm.nih.gov":
        pmids = list(set(extract_pmids_from_pubmed_email(message)))
        logger.info("Found {} PMIDs in PubMed email", len(pmids))
        articles.extend(
            request_articles_from_ncbi_entrez(pmids, contact_email=contact_email)
        )
        logger.info(
            "Found {} articles from {} PMIDs in PubMed email", len(articles), len(pmids)
        )
        return articles
    if from_ == "Nature@e-alert.nature.com" or from_ == "ealert@nature.com":
        match from_:
            case "Nature@e-alert.nature.com":  # emails that are from "Nature Alert"
                dois = list(set(extract_dois_from_nature_alert_email(message)))
            case "ealert@nature.com":  # emails that are from "Nature eAlerts"
                dois = list(set(extract_dois_from_nature_ealerts_email(message)))
        # try semantic scholar
        articles.extend(request_articles_from_semantic_scholar(dois))
        missing_dois = set(dois) - set([article.doi for article in articles])
        # try crossref
        articles.extend(request_articles_from_crossref(missing_dois))
        missing_dois = set(dois) - set([article.doi for article in articles])
        # try Nature pages directly
        for doi in missing_dois:
            nature_id = doi.split("/")[-1]
            url = f"https://www.nature.com/articles/{nature_id}"
            try:
                articles.append(request_article_from_nature_url(url))
            except Exception:
                logger.debug("Failed to get article metadata from Nature: {}", doi)
                continue
        return articles
    if from_ == "do-not-reply@semanticscholar.org":
        ids = list(set(extract_ids_from_semantic_scholar_email(message)))
        articles.extend(request_articles_from_semantic_scholar(ids))
        return articles
    if from_ == "no-reply@arXiv.org":
        ids = list(set(extract_ids_from_arxiv_email(message)))
        articles.extend(request_articles_from_arxiv(ids))
        return articles
    if from_ == "alerts@aaas.sciencepubs.org":
        urls = extract_urls_from_science_email(message)
        for url in urls:
            try:
                articles.append(request_article_from_science_url(url))
            except Exception:
                logger.debug("Failed to get article metadata from Science: {}", url)
                continue
        return articles
    # not implemented below
    # elif from_ in ["scholaralerts-noreply@google.com", "scholarcitations-noreply@google.com"]:
    #     urls = parse_email_google_scholar(message)
    # elif from_ == "oxfordacademicalerts@oup.com":
    #     return parse_email_oxford_academic(message)
    
    # elif from_ == "cellpress@notification.elsevier.com":
    #     return parse_email_cellpress(message)
    # elif from_ == "nejmtoc@n.nejm.org":
    #     return parse_email_nejm(message)
    # elif from_ == "busybee@blogtrottr.com" and "BioDecoded" in subject:
    #     return parse_email_biodecoded(message)
    # elif from_ == "announce@annualreviews.org":
    #     return parse_email_annual_reviews(message)

    logger.warning("No email handler for sender: {}", from_)
    raise NotImplementedError(f"Unknown sender: {from_}")


def generate_report(
    ref_df: pl.DataFrame, query_df: pl.DataFrame, threshold: float
) -> str:
    fig, ax = plt.subplots()
    sns.kdeplot(ref_df, x="cv_distance", hue="cv_fold", ax=ax, legend=False)
    for distance in query_df.get_column("distance").to_list():
        ax.axvline(x=distance, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    num_matches = query_df.filter(pl.col("distance") < threshold).shape[0]
    num_articles = query_df.shape[0]
    percent_matches = num_matches / num_articles * 100

    # generate HTML report content
    report_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Matching Papers</title>
    </head>
    <body>
        <h1>Matching Papers</h1>
        <p>{num_matches} out of {num_articles} matched</p>
        <p>{percent_matches:.1f}% of papers matched</p>
        <p>Threshold: {threshold:.3f}</p>
        <br>
    """
    for article in (
        query_df.filter(pl.col("distance") < threshold)
        .sort("distance")
        .iter_rows(named=True)
    ):
        article_url = (
            f"https://doi.org/{article['doi']}"
            if article["journal"] != "arXiv"
            else f"https://arxiv.org/abs/{article['arxiv']}"
        )
        report_html += f"""
        <div style='margin: 1em 0;'>
            <strong><a href='{article_url}'>{article["title"]}</a></strong><br>
            {article["authors"]}<br>
            <em>{article["journal"]}</em><br>
            <a href='{article_url}'>{article_url}</a><br>
            <span style='color: #666;'>Distance: {article["distance"]:.3f}</span>
        </div>
        """
    report_html += f"""
        <div style='margin: 2em 0;'>
            <h2>Distance Distribution</h2>
            <img src="data:image/png;base64,{plot_base64}"
                    style="max-width: 100%; height: auto;"
                    alt="Distance Distribution Plot">
        </div>
        <div style='margin-top: 3em; padding: 1em 0; border-top: 1px solid #eee; color: #999; font-size: 0.8em; text-align: center;'>
            LEIBLEREPORT
        </div>
    </body>
    </html>
    """
    return report_html


def construct_report_email(
    original_message: email.message.EmailMessage, report_html: str, user: str
) -> email.message.EmailMessage:
    email.policy.SMTP.header_factory.registry["references"] = (
        email.headerregistry.MessageIDHeader
    )
    email.policy.SMTP.header_factory.registry["in-reply-to"] = (
        email.headerregistry.MessageIDHeader
    )
    response = email.message.EmailMessage(policy=email.policy.SMTP)
    response["Subject"] = "Fwd: " + original_message.get("Subject")
    response["From"] = user
    response["To"] = user
    response["Date"] = email.utils.formatdate(localtime=True)
    response["Message-ID"] = email.utils.make_msgid(domain=user.split("@")[1])
    response.add_header("References", original_message.get("Message-ID"))
    response.add_header("In-Reply-To", original_message.get("Message-ID"))
    response.add_header("X-Forwarded-For", original_message.get("From"))
    response.set_content(
        "This is an HTML email. Please use an HTML-capable email client."
    )
    response.add_alternative(report_html, subtype="html")
    return response
