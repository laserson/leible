import torch
from adapters import AutoAdapterModel
from joblib import Parallel, delayed
from openai import OpenAI
from ratelimit import limits, sleep_and_retry
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from leible.models import Article, Embedding


@sleep_and_retry
@limits(calls=1, period=1)
def embed_articles_openai(
    client: OpenAI, model: str, articles: list[Article]
) -> list[Embedding]:
    assert len(articles) < 2048, "OpenAI API only supports up to 2048 inputs"
    inputs = [f"{article.title}\n\n{article.abstract}" for article in articles]
    response = client.embeddings.create(input=inputs, model=model)
    return [
        Embedding(
            article_id=article._id, input=input, model=model, embedding=data.embedding
        )
        for article, input, data in zip(articles, inputs, response.data)
    ]


def embed_articles_specter2(
    articles: list[Article], n_jobs: int = 4, batch_size: int = 32
) -> list[Embedding]:
    if len(articles) == 0:
        return []

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2", source="hf", load_as="specter2", set_active=True
    )

    dataset = [
        f"{article.title}{tokenizer.sep_token}{article.abstract}"
        for article in articles
    ]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def embed_batch(batch):
        with torch.no_grad():
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
            return batch_embeddings

    embeddings = Parallel(n_jobs=n_jobs)(
        delayed(embed_batch)(batch) for batch in dataloader
    )
    embeddings = torch.cat(embeddings, dim=0)
    return [
        Embedding(
            article_id=article._id,
            input=input,
            model="specter2",
            embedding=embedding.tolist(),
        )
        for article, input, embedding in zip(articles, dataset, embeddings)
    ]
