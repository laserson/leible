# leible

```bash
READCUBE_EXPORT_CSV=$HOME/proj/leible/data/readcube_papers_export.csv
cat $READCUBE_EXPORT_CSV | duckdb -csv -c "SELECT * FROM read_csv('/dev/stdin') WHERE \"created (Read-Only)\" >= '2022-01-01'" > readcube_filtered.csv
leible readcube --readcube-csv readcube_filtered.csv --output-parquet readcube_embedded.parquet
leible emails --positives-parquet readcube_embedded.parquet --threshold 0.08
```
    df = pl.read_csv(csv_path, infer_schema=False).filter(
        pl.col("created (Read-Only)") > "2022-01-01"
    )
    logger.info("Attempting to load {} papers from ReadCube", len(df))



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
