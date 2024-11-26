# leible


```
# Train model on library of articles
dois: list[str]
=> articles: list[Article]
=> embeddings: list[Embedding]
=> knn: NearestNeighbors

# Classify new articles based on distance to library
candidates: list[Article]
=> embeddings: list[Embedding]
=> distances: list[float]
```

```
# .env
LEIBLE_IMAP_SERVER="imap.example.com"
LEIBLE_IMAP_PORT=993
LEIBLE_IMAP_USER="user@example.com"
LEIBLE_IMAP_PASS="MySecretPassword"
LEIBLE_IMAP_FOLDER_QUEUE="ToC"
LEIBLE_IMAP_FOLDER_PROCESSED="ToC-processed"
LEIBLE_SMTP_HOST="smtp.example.com"
LEIBLE_SMTP_PORT=465
LEIBLE_SMTP_USER="user@example.com"
LEIBLE_SMTP_PASS="MySecretPassword"
```
