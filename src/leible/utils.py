from itertools import filterfalse
from pathlib import Path
from typing import Callable

import requests
from dotenv import load_dotenv
from loguru import logger
from playwright.sync_api import sync_playwright
from ratelimit import limits, sleep_and_retry

from leible.models import Article


@sleep_and_retry
@limits(calls=5, period=1)
def doi_to_redirected_url(doi: str) -> str:
    response = requests.head(f"https://doi.org/{doi}", allow_redirects=True)
    return response.url


def partition_by_predicate(
    pred: Callable[[Article], bool], xs: list[Article]
) -> tuple[list[Article], list[Article]]:
    return list(filter(pred, xs)), list(filterfalse(pred, xs))


def get_first_simple_type(x):
    if x is None or isinstance(x, (int, float, str, bool)):
        return x
    return get_first_simple_type(next(iter(x)))


def load_config() -> None:
    env_file = Path.home() / ".config" / "leible" / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        load_dotenv()
        logger.info("Loaded environment variables from CWD .env")


def get_page_content_with_playwright(url: str) -> str:
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_selector("main", timeout=10000)
        content = page.content()
        browser.close()
        return content
