import os
import smtplib
import sys
from pathlib import Path

import numpy as np
import polars as pl
from click import Choice, group, option
from click import Path as ClickPath
from dotenv import load_dotenv
from imapclient import IMAPClient
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from leible.email import (
    construct_report_email,
    fetch_emails,
    generate_report,
    process_email,
)
from leible.embeddings import embed_articles_specter2
from leible.utils import load_readcube_papers_csv


@group()
def cli():
    pass


@cli.command()
@option(
    "--library-csv",
    type=ClickPath(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@option("--threshold", type=float, default=0.08)
@option(
    "--log-level",
    type=Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def emails(library_csv: Path, threshold: float, log_level: str):
    env_file = Path.home() / ".config" / "leible" / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        load_dotenv()
        logger.info("Loaded environment variables from CWD .env")
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    emails = fetch_emails(
        os.environ["LEIBLE_IMAP_SERVER"],
        os.environ["LEIBLE_IMAP_PORT"],
        os.environ["LEIBLE_IMAP_USER"],
        os.environ["LEIBLE_IMAP_PASS"],
        os.environ["LEIBLE_IMAP_FOLDER_QUEUE"],
    )
    logger.info(f"Fetched {len(emails)} emails for processing")

    # load library and embed articles
    articles = load_readcube_papers_csv(library_csv, os.environ["LEIBLE_IMAP_USER"])
    logger.info(f"Loaded {len(articles)} articles from {library_csv}")
    embeddings = embed_articles_specter2(articles)
    embeddings = np.asarray([e.embedding for e in embeddings])
    logger.info(f"Embedded {len(embeddings)} articles")

    # run 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_validation_stats = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
        knn = NearestNeighbors(n_neighbors=10, metric="cosine")
        knn.fit(embeddings[train_idx])
        distances, _ = knn.kneighbors(embeddings[test_idx])
        cross_validation_stats.append(
            pl.DataFrame({"fold": str(fold), "distance": distances.mean(axis=1)})
        )
    cross_validation_stats = pl.concat(cross_validation_stats)
    logger.info("Computed 5-fold cross validation distances")

    # generate the k-NN graph for the library
    knn = NearestNeighbors(n_neighbors=10, metric="cosine")
    knn.fit(embeddings)
    logger.info("Created k-NN graph for classification")

    # process emails
    for message in emails:
        logger.info("Starting to process email {}", message.get("Subject"))

        try:
            articles = process_email(message)
        except NotImplementedError:
            logger.info("Email from {} not implemented", message.get("From"))
            continue
        if len(articles) == 0:
            logger.warning(
                "Email {} from {} returned no articles",
                message.get("Subject"),
                message.get("From"),
            )
            logger.debug("Email content:\n{}", str(message))
            continue
        embeddings = embed_articles_specter2(articles)
        embeddings = np.asarray([e.embedding for e in embeddings])
        distances, _ = knn.kneighbors(embeddings)
        logger.info(f"Embedded and scored {len(articles)} articles")
        articles_df = pl.DataFrame(articles).with_columns(
            pl.Series("distance", distances.mean(axis=1))
        )
        num_matches = articles_df.filter(pl.col("distance") < threshold).shape[0]
        logger.info(f"Found {num_matches} matches out of {len(articles)} articles")

        report_html = generate_report(cross_validation_stats, articles_df, threshold)
        logger.info("Generated HTML report")
        response = construct_report_email(
            message, report_html, os.environ["LEIBLE_SMTP_USER"]
        )
        logger.info("Constructed report email")
        # send the email
        with smtplib.SMTP_SSL(
            os.environ["LEIBLE_SMTP_HOST"], os.environ["LEIBLE_SMTP_PORT"]
        ) as server:
            server.login(os.environ["LEIBLE_SMTP_USER"], os.environ["LEIBLE_SMTP_PASS"])
            server.send_message(response)
        logger.info("Sent report email")

        # Mark the email as processed by moving it to the processed folder
        with IMAPClient(
            host=os.environ["LEIBLE_IMAP_SERVER"],
            port=os.environ["LEIBLE_IMAP_PORT"],
            use_uid=True,
        ) as client:
            client.login(os.environ["LEIBLE_IMAP_USER"], os.environ["LEIBLE_IMAP_PASS"])
            client.select_folder(os.environ["LEIBLE_IMAP_FOLDER_QUEUE"], readonly=False)
            client.move(
                int(message.get("X-IMAP-UID")),
                os.environ["LEIBLE_IMAP_FOLDER_PROCESSED"],
            )
        logger.info(
            "Moved original email to {}", os.environ["LEIBLE_IMAP_FOLDER_PROCESSED"]
        )
        logger.info("Finished processing email {}", message.get("Subject"))
