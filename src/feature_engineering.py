import polars as pl
import logging
from config import FEATURES_PATH, PARQUET_5CORE_PATH
import os

logger = logging.getLogger(__name__)

def extract_behavioral_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Estrae le 14 feature comportamentali e testuali per ogni user_id usando l'engine super-efficiente di Polars.
    """
    logger.info("Iniziando l'estrazione delle feature comportamentali...")
    
    # 1. Parsing timestamp da ms a datetime (se non gia datetime)
    if df.schema["timestamp"] in [pl.Int64, pl.Float64, pl.Int32]:
        df = df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("date"))
    else:
        df = df.with_columns(pl.col("timestamp").alias("date"))
        
    # Preparazione testi ed espressioni nativa Polars
    clean_text = pl.col("text").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    words = clean_text.str.split(" ")
    total_words = words.list.len()
    unique_words = words.list.unique().list.len()
    
    is_empty = clean_text == ""
    
    ttr_expr = pl.when(is_empty).then(0.0).otherwise(unique_words / total_words)
    
    text_len = pl.col("text").fill_null("").str.len_chars()
    clean_len = pl.col("text").fill_null("").str.replace_all(r"[^\w\s]", "").str.len_chars()
    punct_ratio_expr = pl.when(text_len > 0).then((text_len - clean_len) / text_len).otherwise(0.0)
    
    char_count = pl.col("text").fill_null("").str.replace_all(r"\s", "").str.replace_all(r"[^\w]", "").str.len_chars()
    avg_word_len_expr = pl.when(is_empty).then(0.0).otherwise(char_count / total_words)

    df = df.with_columns([
        pl.col("date").dt.date().alias("calendar_day"),
        text_len.alias("text_len"),
        ttr_expr.alias("ttr"),
        avg_word_len_expr.alias("avg_word_len"),
        punct_ratio_expr.alias("punct_ratio")
    ])

    # 2. Computo del 'burstiness': massimo numero di recensioni in un giorno per ciascun utente
    burstiness_df = (
        df.group_by(["user_id", "calendar_day"])
          .len()
          .group_by("user_id")
          .agg(pl.col("len").max().alias("burstiness"))
    )

    # 3. Aggregazioni principali user-level
    user_features = df.group_by("user_id").agg([
        pl.len().alias("review_count"),
        pl.col("rating").mean().alias("avg_rating"),
        pl.col("rating").std().fill_null(0.0).alias("rating_std"), # Std dev is null for n=1
        
        # Pct extreme: % di rating che sono 1.0 o 5.0
        (pl.col("rating").is_in([1.0, 5.0]).sum() / pl.len()).alias("pct_extreme"),
        
        # Pct verified: % di verified_purchase == True
        # Cast to float avoids issues if bool
        (pl.col("verified_purchase").cast(pl.Float64).sum() / pl.len()).alias("pct_verified"),
        
        pl.col("helpful_vote").mean().alias("avg_helpful"),
        
        # Activity span days: delta in giorni tra max e min data
        (pl.col("date").max() - pl.col("date").min()).dt.total_days().alias("activity_span_days"),
        
        pl.col("text_len").mean().alias("avg_text_length"),
        
        # Unique items ratio: Items unici / totale review
        (pl.col("parent_asin").n_unique() / pl.len()).alias("unique_items_ratio"),
        
        # Nuove feature testuali statistiche
        pl.col("ttr").mean().alias("avg_ttr"),
        pl.col("avg_word_len").mean().alias("avg_word_len"),
        pl.col("punct_ratio").mean().alias("avg_punct_ratio")
    ])
    
    # 4. Rating entropy: Approssimiamo la Shannon Entropy sui punteggi (1 a 5)
    # entropy = - sum (p * log2(p))
    rating_counts = df.group_by(["user_id", "rating"]).len()
    user_totals = df.group_by("user_id").len().rename({"len": "total_reviews"})
    
    entropy_df = rating_counts.join(user_totals, on="user_id")
    entropy_df = entropy_df.with_columns(
        (pl.col("len") / pl.col("total_reviews")).alias("p")
    )
    # Computiamo l'entropia log2
    import numpy as np
    entropy_df = entropy_df.with_columns(
        (-pl.col("p") * pl.col("p").log(2)).alias("p_log_p")
    )
    final_entropy = entropy_df.group_by("user_id").agg(
        pl.col("p_log_p").sum().alias("rating_entropy")
    )

    # 5. Join di tutte le componenti
    features_df = (
        user_features
        .join(burstiness_df, on="user_id", how="left")
        .join(final_entropy, on="user_id", how="left")
    )

    # Fill NaN per sicurezza
    features_df = features_df.fill_null(0.0).fill_nan(0.0)
    
    logger.info(f"Estrattele 11 feature per {features_df.height} utenti unici.")
    return features_df

def run_feature_engineering():
    if os.path.exists(FEATURES_PATH):
        logger.info(f"{FEATURES_PATH} esiste già.")
        return
        
    df = pl.read_parquet(PARQUET_5CORE_PATH)
    features = extract_behavioral_features(df)
    features.write_parquet(FEATURES_PATH)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_feature_engineering()
