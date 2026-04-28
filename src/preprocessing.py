import polars as pl
import logging

logger = logging.getLogger(__name__)

def clean_text_columns(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    Applica cleaning minimo (niente stop-words, rimozione HTML, URL, e whitespace)
    e concatena 'title' e 'text' in una nuova colonna 'review_text'.
    Ottimizzato per engine Rust di Polars (no UDF Python per max performance).
    """
    logger.info("Avvio preprocessing testuale su title e text...")
    
    # Regex patterns
    HTML_PATTERN = r"<[^>]*>"
    URL_PATTERN = r"https?://\S+|www\.\S+"
    WHITESPACE_PATTERN = r"\s+"
    
    # Riempiamo i null testuali per evitare propagazione di NaN nella concatenazione
    df_cleaned = df.with_columns([
        pl.col("title").fill_null(""),
        pl.col("text").fill_null("")
    ])
    
    for col_name in ["title", "text"]:
        df_cleaned = df_cleaned.with_columns(
            pl.col(col_name)
            .str.replace_all(HTML_PATTERN, " ")
            .str.replace_all(URL_PATTERN, " ")
            .str.replace_all(WHITESPACE_PATTERN, " ")
            .str.strip_chars()
        )
        
    df_cleaned = df_cleaned.with_columns(
        (pl.col("title") + " " + pl.col("text"))
        .str.replace_all(WHITESPACE_PATTERN, " ")
        .str.strip_chars()
        .alias("review_text")
    )
    
    logger.info("Preprocessing text completato. Aggiunta colonna 'review_text'.")
    return df_cleaned
