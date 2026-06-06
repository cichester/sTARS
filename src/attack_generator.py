import polars as pl
import numpy as np
import logging
from src.genai_client import GeminiClient

logger = logging.getLogger(__name__)

def select_target_item(df: pl.DataFrame) -> str:
    """Seleziona un item target 'medio' dalla fascia mediana di popolarità."""
    item_counts = df.group_by("parent_asin").len().sort("len")
    # Prendiamo un item al 50esimo percentile
    mid_idx = item_counts.height // 2
    target = item_counts["parent_asin"][mid_idx]
    logger.info(f"Target item selezionato: {target} (con {item_counts['len'][mid_idx]} recensioni originali)")
    return target

def generate_average_attack(df: pl.DataFrame, target_item: str, num_bots: int = 50, filler_size: int = 20, seed: int = 42) -> pl.DataFrame:
    """Genera bot con profilo Average Attack."""
    logger.info(f"Generazione Average Attack per il target {target_item} con {num_bots} bot...")
    np.random.seed(seed)
    
    # Statistiche globali
    global_mean = df["rating"].mean()
    global_std = df["rating"].std()
    
    bot_records = []
    
    # Campioniamo item casuali per i filler
    unique_items = df["parent_asin"].unique().to_list()
    
    for i in range(num_bots):
        bot_id = f"BOT_AVG_{i:04d}"
        
        # Target item
        bot_records.append({
            "rating": 5.0, # Push attack
            "title": "Amazing product!",
            "text": "This is exactly what I was looking for. Highly recommended.",
            "images": [],
            "asin": target_item,
            "parent_asin": target_item,
            "user_id": bot_id,
            "timestamp": int(np.random.uniform(1.6e12, 1.7e12)), # Random timestamp in ms
            "helpful_vote": 0,
            "verified_purchase": False
        })
        
        # Filler items
        available_fillers = [item for item in unique_items if item != target_item]
        num_to_sample = min(filler_size, len(available_fillers))
        filler_items = np.random.choice(available_fillers, size=num_to_sample, replace=False)
        for f_item in filler_items:
            
            # Rating vicino alla media globale
            f_rating = np.clip(np.random.normal(global_mean, global_std), 1.0, 5.0)
            f_rating = float(round(f_rating))
            
            bot_records.append({
                "rating": f_rating,
                "title": "Good",
                "text": "It works as expected, nothing special but does the job.",
                "images": [],
                "asin": f_item,
                "parent_asin": f_item,
                "user_id": bot_id,
                "timestamp": int(np.random.uniform(1.6e12, 1.7e12)),
                "helpful_vote": 0,
                "verified_purchase": False
            })
            
    bot_df = pl.DataFrame(bot_records)
    
    # Rinominiamo la colonna text in review_text per compatibilità con il resto della pipeline
    if "text" in bot_df.columns:
        bot_df = bot_df.with_columns(pl.col("text").alias("review_text"))
        
    return bot_df

def generate_bandwagon_attack(df: pl.DataFrame, target_item: str, num_bots: int = 50, filler_size: int = 20, seed: int = 42) -> pl.DataFrame:
    """Genera bot con profilo Bandwagon Attack."""
    logger.info(f"Generazione Bandwagon Attack per il target {target_item} con {num_bots} bot...")
    np.random.seed(seed + 1)
    
    # Identifichiamo gli item più popolari
    item_counts = df.group_by("parent_asin").len().sort("len", descending=True)
    top_items = item_counts.head(200)["parent_asin"].to_list()
    
    bot_records = []
    
    for i in range(num_bots):
        bot_id = f"BOT_BW_{i:04d}"
        
        # Target item
        bot_records.append({
            "rating": 5.0, # Push attack
            "title": "Perfect!",
            "text": "I absolutely love this product. Five stars.",
            "images": [],
            "asin": target_item,
            "parent_asin": target_item,
            "user_id": bot_id,
            "timestamp": int(np.random.uniform(1.6e12, 1.7e12)),
            "helpful_vote": 0,
            "verified_purchase": False
        })
        
        # Filler items (presi dai top_items)
        available_fillers = [item for item in top_items if item != target_item]
        num_to_sample = min(filler_size, len(available_fillers))
        filler_items = np.random.choice(available_fillers, size=num_to_sample, replace=False)
        for f_item in filler_items:
            
            # Rating alto per items popolari
            f_rating = float(np.random.choice([4.0, 5.0]))
            
            bot_records.append({
                "rating": f_rating,
                "title": "Great",
                "text": "Very popular and for a good reason. Works great.",
                "images": [],
                "asin": f_item,
                "parent_asin": f_item,
                "user_id": bot_id,
                "timestamp": int(np.random.uniform(1.6e12, 1.7e12)),
                "helpful_vote": 0,
                "verified_purchase": False
            })
            
    bot_df = pl.DataFrame(bot_records)
    
    if "text" in bot_df.columns:
        bot_df = bot_df.with_columns(pl.col("text").alias("review_text"))
        
    return bot_df


def generate_genai_attack(df: pl.DataFrame, target_item: str, num_bots: int = 50, filler_size: int = 20, seed: int = 42) -> pl.DataFrame:
    """
    Genera bot con profilo GenAI Attack.
    I testi del target e dei filler vengono generati tramite l'API di Google Gemini (o recuperati da cache).
    Il comportamento simula un attacco ibrido:
    - Target: recensito con rating 5.0 e testo positivo generato da GenAI.
    - Filler: mix del 50% di item popolari e 50% di item casuali, con rating realistici e testi generati da GenAI.
    """
    logger.info(f"Generazione GenAI Attack per il target {target_item} con {num_bots} bot...")
    np.random.seed(seed + 2)
    
    client = GeminiClient()
    
    # 1. Generazione di num_bots recensioni per il target (tutte positive)
    logger.info(f"Generazione dei testi GenAI per l'item target {target_item}...")
    target_reviews = client.generate_reviews(target_item, num_bots, "positive")
    
    # 2. Generazione del pool di recensioni generiche per i filler per evitare migliaia di API call lente
    logger.info("Generazione del pool di recensioni generiche per i filler...")
    filler_positives = client.generate_reviews("generic_filler", 50, "positive")
    filler_neutrals = client.generate_reviews("generic_filler", 50, "neutral")

    # Statistiche globali
    item_counts = df.group_by("parent_asin").len().sort("len", descending=True)
    top_items = item_counts.head(200)["parent_asin"].to_list()
    all_items = df["parent_asin"].unique().to_list()
    
    bot_records = []
    
    for i in range(num_bots):
        bot_id = f"BOT_GENAI_{i:04d}"
        
        # Target review (usiamo a rotazione quelle generate)
        t_rev = target_reviews[i % len(target_reviews)]
        bot_records.append({
            "rating": 5.0, # Push attack
            "title": t_rev["title"],
            "text": t_rev["text"],
            "images": [],
            "asin": target_item,
            "parent_asin": target_item,
            "user_id": bot_id,
            "timestamp": int(np.random.uniform(1.6e12, 1.7e12)),
            "helpful_vote": 0,
            "verified_purchase": False
        })
        
        # Selezione dei filler items (ibrido 50% popolari, 50% random)
        fillers = []
        
        # Escludiamo il target dai possibili filler
        available_top = [item for item in top_items if item != target_item]
        num_top = min(filler_size // 2, len(available_top))
        if num_top > 0:
            fillers.extend(np.random.choice(available_top, size=num_top, replace=False))
            
        available_all = [item for item in all_items if item != target_item and item not in fillers]
        num_all = min(filler_size - len(fillers), len(available_all))
        if num_all > 0:
            fillers.extend(np.random.choice(available_all, size=num_all, replace=False))
            
        # Assegniamo testi dal pool di recensioni generiche per ciascun filler item di questo bot
        for f_item in fillers:
            # Assegniamo un sentiment casuale (neutral o positive per i filler)
            f_sentiment = np.random.choice(["neutral", "positive"])
            f_rating = 5.0 if f_sentiment == "positive" else 4.0 if f_sentiment == "neutral" else 3.0
            
            # Peschiamo una recensione casuale dal pool corretto
            pool = filler_positives if f_sentiment == "positive" else filler_neutrals
            f_rev = np.random.choice(pool) if pool else {"title": "Good", "text": "It works as expected for this product."}
            
            bot_records.append({
                "rating": float(f_rating),
                "title": f_rev["title"],
                "text": f_rev["text"],
                "images": [],
                "asin": f_item,
                "parent_asin": f_item,
                "user_id": bot_id,
                "timestamp": int(np.random.uniform(1.6e12, 1.7e12)),
                "helpful_vote": 0,
                "verified_purchase": False
            })
            
    bot_df = pl.DataFrame(bot_records)
    
    if "text" in bot_df.columns:
        bot_df = bot_df.with_columns(pl.col("text").alias("review_text"))
        
    return bot_df

def inject_bots(clean_df: pl.DataFrame, bot_df: pl.DataFrame) -> pl.DataFrame:
    """Concatena il dataset originale con i record dei bot."""
    logger.info(f"Iniezione di {bot_df.height} recensioni bot nel dataset originale...")
    
    # Assicuriamoci che gli schema combacino
    common_cols = [c for c in clean_df.columns if c in bot_df.columns]
    
    # Eseguiamo il cast e il concat
    poisoned_df = pl.concat([clean_df.select(common_cols), bot_df.select(common_cols)], how="vertical")
    
    return poisoned_df
