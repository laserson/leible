import base64
import email
import email.policy
import io
import re

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from bs4 import BeautifulSoup
from imapclient import IMAPClient
from loguru import logger

from leible.metadata import get_article_from_biorxiv, get_articles_by_pmids
from leible.models import Article


def fetch_emails(
    imap_host, imap_port, imap_user, imap_pass, folder="ToC"
) -> list[email.message.EmailMessage]:
    with IMAPClient(host=imap_host, port=imap_port, use_uid=True) as client:
        client.login(imap_user, imap_pass)
        client.select_folder(folder, readonly=True)
        message_ids = client.search()
        emails = []
        for message_uid, message_data in client.fetch(message_ids, [b"RFC822", b"FLAGS"]).items():
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


def extract_urls_from_biorxiv_html(soup: BeautifulSoup) -> list[str]:
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


def parse_email_biorxiv(message: email.message.EmailMessage) -> list[Article]:
    subject = message.get("Subject")
    soup = extract_html_from_email(message)
    urls = extract_urls_from_biorxiv_html(soup)
    articles = []
    for url in urls:
        try:
            article = get_article_from_biorxiv(url)
        except Exception as e:
            logger.error("Failed to load article: {}", url)
            continue
        article.source = f"BioRxiv {subject}"
        articles.append(article)
    return articles


def parse_email_google_scholar(message: email.message.EmailMessage) -> list[Article]:
    raise NotImplementedError("Google Scholar is not implemented")


def parse_email_pubmed(message: email.message.EmailMessage) -> list[Article]:
    soup = extract_html_from_email(message)
    anchors = soup.find_all("a", class_="docsum-title")
    pmids = [re.fullmatch(r"article_id=(\d+)", a.get("ref")).group(1) for a in anchors]
    articles = get_articles_by_pmids(pmids)
    for article in articles:
        article.source = "NCBI PubMed alert"
    return articles


def parse_email_nature(message: email.message.EmailMessage) -> list[str]:
    raise NotImplementedError("Nature is not implemented")


def parse_email_science(message: email.message.EmailMessage) -> list[str]:
    raise NotImplementedError("Science is not implemented")


def parse_email_cellpress(message: email.message.EmailMessage) -> list[str]:
    raise NotImplementedError("Cell Press is not implemented")


def parse_email_nejm(message: email.message.EmailMessage) -> list[str]:
    raise NotImplementedError("NEJM is not implemented")


def parse_email_annual_reviews(message: email.message.EmailMessage) -> list[str]:
    raise NotImplementedError("Annual Reviews is not implemented")


def parse_email_oxford_academic(message: email.message.EmailMessage) -> list[str]:
    raise NotImplementedError("Oxford Academic is not implemented")


def parse_email_biodecoded(message: email.message.EmailMessage) -> list[str]:
    raise NotImplementedError("BioDecoded is not implemented")


def parse_email(message: email.message.EmailMessage) -> list[Article]:
    _, from_ = email.utils.parseaddr(message.get("From"))
    subject = message.get("Subject")

    match from_:
        case "cshljnls-mailer@alerts.highwire.org":
            return parse_email_biorxiv(message)
        case "scholaralerts-noreply@google.com" | "scholarcitations-noreply@google.com":
            return parse_email_google_scholar(message)
        case "efback@ncbi.nlm.nih.gov":
            return parse_email_pubmed(message)
        case "ealert@nature.com":
            return parse_email_nature(message)
        case "oxfordacademicalerts@oup.com":
            return parse_email_oxford_academic(message)
        case "alerts@aaas.sciencepubs.org":
            return parse_email_science(message)
        case "cellpress@notification.elsevier.com":
            return parse_email_cellpress(message)
        case "nejmtoc@n.nejm.org":
            return parse_email_nejm(message)
        case "busybee@blogtrottr.com" if "BioDecoded" in subject:
            return parse_email_biodecoded(message)
        case "announce@annualreviews.org":
            return parse_email_annual_reviews(message)

    logger.warning("Unhandled email from: {}", from_)
    return []


def generate_report(
    distance_stats: pl.DataFrame, articles_df: pl.DataFrame, threshold: float
) -> str:
    fig, ax = plt.subplots()
    sns.kdeplot(distance_stats, x="distance", hue="fold", ax=ax, legend=False)
    for distance in articles_df.get_column("distance").to_list():
        ax.axvline(x=distance, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    num_matches = articles_df.filter(pl.col("distance") < threshold).shape[0]
    num_articles = articles_df.shape[0]
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
        articles_df.filter(pl.col("distance") < threshold)
        .sort("distance")
        .iter_rows(named=True)
    ):
        report_html += f"""
        <div style='margin: 1em 0;'>
            <strong>{article["title"]}</strong><br>
            <em>{article["journal"]}</em><br>
            <a href='https://doi.org/{article["doi"]}'>https://doi.org/{article["doi"]}</a><br>
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
