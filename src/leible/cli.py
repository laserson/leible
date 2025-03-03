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
from tqdm import tqdm

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
    load_specter2_model,
)
from leible.metadata import simple_load_readcube_papers_csv
from leible.utils import load_config


@group()
def cli():
    pass


@cli.command()
@option(
    "--readcube-csv",
    type=ClickPath(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@option(
    "--output-parquet",
    type=ClickPath(exists=False, dir_okay=False, path_type=Path),
    required=True,
)
def readcube(readcube_csv: Path, output_parquet: Path):
    """Load, embed, and cross-validate ReadCube papers."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    load_config()

    readcube_articles = simple_load_readcube_papers_csv(readcube_csv)
    readcube_df = pl.DataFrame(readcube_articles)
    readcube_df = embed_articles_specter2(readcube_df)
    readcube_embeddings = np.asarray(readcube_df.get_column("embedding").to_list())
    # run cross validation
    folds, distances = compute_cross_validation_stats(readcube_embeddings)
    readcube_df = readcube_df.with_columns(
        pl.Series("cv_fold", folds), pl.Series("cv_distance", distances)
    )
    readcube_df.write_parquet(output_parquet)
    logger.info(f"Saved {len(readcube_df)} embedded articles to {output_parquet}")


@cli.command()
@option(
    "--positives-parquet",
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
def emails(positives_parquet: Path, threshold: float, log_level: str):
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    load_config()

    # load pre-embedded positives
    positives_df = pl.read_parquet(positives_parquet)
    positives_embeddings = np.asarray(positives_df.get_column("embedding").to_list())

    # load SPECTER2 model for embedding articles from emails
    model, tokenizer = load_specter2_model()

    # fetch emails from IMAP
    emails = fetch_emails(
        os.environ["LEIBLE_IMAP_SERVER"],
        os.environ["LEIBLE_IMAP_PORT"],
        os.environ["LEIBLE_IMAP_USER"],
        os.environ["LEIBLE_IMAP_PASS"],
        os.environ["LEIBLE_IMAP_FOLDER_QUEUE"],
    )
    logger.info(f"Fetched {len(emails)} emails for processing")

    for message in tqdm(emails, desc="Processing emails"):
        logger.info("Starting to process email {}", message.get("Subject"))

        try:
            query_articles = process_email(
                message,
                contact_email=os.environ["LEIBLE_IMAP_USER"],
                s2_api_key=os.environ.get("LEIBLE_S2_API_KEY"),
            )
        except NotImplementedError:
            logger.info("Email from {} not implemented", message.get("From"))
            continue
        if len(query_articles) == 0:
            logger.warning(
                "Email {} from {} returned no articles",
                message.get("Subject"),
                message.get("From"),
            )
            logger.debug(f"Email content:\n{str(message)}")
            continue
        # remove articles with missing title
        query_df = pl.DataFrame(query_articles).filter(pl.col("title").is_not_null())
        if len(query_df) == 0:
            logger.warning(
                "Email {} from {} returned {} articles but all are missing title",
                message.get("Subject"),
                message.get("From"),
                len(query_articles),
            )
            logger.debug(f"Email content:\n{str(message)}")
            continue

        query_df = embed_articles_specter2(query_df, model, tokenizer)
        logger.info(f"Loaded and embedded {len(query_df)} articles from email")
        query_embeddings = np.asarray(query_df.get_column("embedding").to_list())
        distances = compute_knn_distances(query_embeddings, positives_embeddings)
        query_df = pl.DataFrame(query_df).with_columns(pl.Series("distance", distances))
        num_matches = query_df.filter(pl.col("distance") < threshold).shape[0]
        logger.info(f"Found {num_matches} matches out of {len(positives_df)} articles")

        report_html = generate_report(
            # drop embedding col bc it's problematic when converted to pandas
            # for seaborn
            positives_df.drop("embedding"),
            query_df.drop("embedding"),
            threshold,
        )
        response_email = construct_report_email(
            message, report_html, os.environ["LEIBLE_SMTP_USER"]
        )
        # send the email
        with smtplib.SMTP_SSL(
            os.environ["LEIBLE_SMTP_HOST"], os.environ["LEIBLE_SMTP_PORT"]
        ) as server:
            server.login(os.environ["LEIBLE_SMTP_USER"], os.environ["LEIBLE_SMTP_PASS"])
            server.send_message(response_email)
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
