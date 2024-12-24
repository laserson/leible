import os
import smtplib
import sys
from pathlib import Path

import numpy as np
import polars as pl
from click import Choice, group, option
from click import Path as ClickPath
from imapclient import IMAPClient
from loguru import logger

from leible.email import (
    construct_report_email,
    fetch_emails,
    generate_report,
    process_email,
)
from leible.embeddings import (
    compute_cross_validation_stats,
    compute_knn_distances,
    embed_articles_specter2,
)
from leible.metadata import load_readcube_papers_csv
from leible.utils import load_config


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
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    load_config()

    emails = fetch_emails(
        os.environ["LEIBLE_IMAP_SERVER"],
        os.environ["LEIBLE_IMAP_PORT"],
        os.environ["LEIBLE_IMAP_USER"],
        os.environ["LEIBLE_IMAP_PASS"],
        os.environ["LEIBLE_IMAP_FOLDER_QUEUE"],
    )
    logger.info(f"Fetched {len(emails)} emails for processing")

    # load reference library and embed articles
    ref_articles = load_readcube_papers_csv(library_csv, os.environ["LEIBLE_IMAP_USER"])
    logger.info(f"Loaded {len(ref_articles)} articles from {library_csv}")
    ref_df = pl.DataFrame(ref_articles)
    ref_df = embed_articles_specter2(ref_df)
    ref_embeddings = np.asarray(ref_df.get_column("embedding").to_list())
    logger.info(f"Embedded {len(ref_df)} articles")

    # run cross validation
    folds, distances = compute_cross_validation_stats(ref_embeddings)
    ref_df = ref_df.with_columns(
        pl.Series("cv_fold", folds), pl.Series("cv_distance", distances)
    )
    logger.info("Computed cross validation distances")

    # process emails
    for message in emails:
        logger.info("Starting to process email {}", message.get("Subject"))

        try:
            query_articles = process_email(message)
        except NotImplementedError:
            logger.info("Email from {} not implemented", message.get("From"))
            continue

        if len(query_articles) == 0:
            logger.warning(
                "Email {} from {} returned no articles",
                message.get("Subject"),
                message.get("From"),
            )
            logger.debug("Email content:\n{}", str(message))
            continue

        # remove articles with missing title or abstract
        query_df = pl.DataFrame(query_articles).filter(
            pl.col("title").is_not_null() & pl.col("abstract").is_not_null()
        )
        if len(query_df) == 0:
            logger.warning(
                "Email {} from {} returned {} articles but all are missing title or abstract",
                message.get("Subject"),
                message.get("From"),
                len(query_articles),
            )
            logger.debug("Email content:\n{}", str(message))
            continue
        query_df = embed_articles_specter2(query_df)
        query_embeddings = np.asarray(query_df.get_column("embedding").to_list())
        distances = compute_knn_distances(query_embeddings, ref_embeddings)
        logger.info(f"Embedded and scored {len(query_df)} articles")
        query_df = pl.DataFrame(query_df).with_columns(pl.Series("distance", distances))
        num_matches = query_df.filter(pl.col("distance") < threshold).shape[0]
        logger.info(f"Found {num_matches} matches out of {len(ref_articles)} articles")

        report_html = generate_report(
            # drop embedding col bc it's problematic when converted to pandas
            # for seaborn
            ref_df.drop("embedding"),
            query_df.drop("embedding"),
            threshold,
        )
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
