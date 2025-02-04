import numpy as np
import polars as pl
import torch
from adapters import AutoAdapterModel
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_specter2_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the SPECTER2 model and tokenizer.

    Returns
    -------
    tuple[PreTrainedModel, PreTrainedTokenizerBase]
        The SPECTER2 model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2", source="hf", load_as="specter2", set_active=True
    )
    return model, tokenizer


def embed_articles_specter2(
    articles_df: pl.DataFrame,
    model: PreTrainedModel = None,
    tokenizer: PreTrainedTokenizerBase = None,
) -> pl.DataFrame:
    """Embed articles using Specter2.

    Parameters
    ----------
    articles_df : pl.DataFrame
        A Polars DataFrame containing article metadata. Must have columns:
            - title : str
                The article title
            - abstract : str
                The article abstract
    model : PreTrainedModel, optional
        The SPECTER2 model to use for embedding. If not provided, a new model will be loaded.
    tokenizer : PreTrainedTokenizerBase, optional
        The tokenizer to use for embedding. If not provided, a new tokenizer will be loaded.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an additional 'embedding' column containing
        the SPECTER2 embeddings as numpy arrays

    Notes
    -----
    Uses the SPECTER2 model from Allen AI to generate embeddings for scientific articles.
    The embeddings are generated from the concatenated title and abstract text.

    Accepts a pre-loaded model and tokenizer so you only need to load it once.
    """
    if model is None:
        model, tokenizer = load_specter2_model()

    embedded_df = articles_df.with_columns(
        pl.col("abstract").fill_null("")
    ).with_columns(
        pl.format(
            "{}{}{}",
            pl.col("title"),
            pl.lit(tokenizer.sep_token),
            pl.col("abstract"),
        ).alias("input")
    )
    dataset = embedded_df.get_column("input").to_list()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            )
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    embedded_df = embedded_df.with_columns(pl.Series(embeddings).alias("embedding"))
    return embedded_df


def compute_knn_distances(
    query_embeddings: np.ndarray, reference_embeddings: np.ndarray
) -> np.ndarray:
    knn = NearestNeighbors(n_neighbors=10, metric="cosine")
    knn.fit(reference_embeddings)
    distances, _ = knn.kneighbors(query_embeddings)
    return distances.mean(axis=1)


def compute_cross_validation_stats(
    embeddings: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = np.zeros(len(embeddings))
    distances = np.zeros(len(embeddings))
    for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
        fold_distances = compute_knn_distances(
            embeddings[test_idx], embeddings[train_idx]
        )
        folds[test_idx] = fold
        distances[test_idx] = fold_distances
    return folds, distances
