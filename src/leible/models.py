from dataclasses import dataclass


@dataclass
class Article:
    title: str
    abstract: str
    journal: str = None
    year: int = None
    source: str = None  # e.g., BioRxiv email, Papers app
    date_added: str = None  # YYYY-MM-DD
    notes: str = None  # extra info, preferably in JSON
    # one of these ids _should_ be set
    doi: str = None
    pmid: str = None
    arxiv: str = None
    # internal id
    _id: int = None


@dataclass
class Embedding:
    input: str  # the input text that was embedded
    model: str  # the name of the model
    embedding: str  # JSON list of floats (as string)
    article_id: int = None  # foreign key to articles _id
