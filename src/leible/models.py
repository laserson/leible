from dataclasses import dataclass


@dataclass
class Article:
    # identifiers
    doi: str = None
    pmid: str = None
    arxiv: str = None
    semantic_scholar: str = None  # Semantic Scholar internal ID (hex)
    publisher_url: str = None  # URL of article on publisher's website
    # metadata
    title: str = None
    abstract: str = None
    journal: str = None
    year: int = None
    authors: str = None  # string containing author info
    notes: str = None  # extra info, preferably in JSON
    embedding: str = None  # string encoding a JSON list of floats
