import polars as pl
import os
import sys

from config import FEATURES_PATH, DATA_DIR

def extract_suspicious():
    features = pl.read_parquet(FEATURES_PATH)
    # Prendiamo i 20 utenti con il trust_score più basso
    worst_users = features.sort("trust_score").head(20)["user_id"].to_list()
    
    clean_path = DATA_DIR / "electronics_5core_clean.parquet"
    if not clean_path.exists():
        print(f"File {clean_path} non trovato.")
        sys.exit(1)
        
    df = pl.read_parquet(clean_path)
    
    # Filtriamo le recensioni per questi 20 utenti
    suspicious_reviews = df.filter(pl.col("user_id").is_in(worst_users))
    
    # Per avere esattamente 20 recensioni uniche (1 per utente peggiore), 
    # prendiamo la prima recensione di ciascuno dei 20 utenti
    res = suspicious_reviews.group_by("user_id").first()
    
    # Aggiungiamo il trust_score per ordinarli
    res = res.join(features.select(["user_id", "trust_score"]), on="user_id").sort("trust_score")
    
    out_file = "results/suspicious_reviews.md"
    os.makedirs("results", exist_ok=True)
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# Le 20 Recensioni Più Sospette\n\n")
        f.write("Queste recensioni appartengono agli utenti con il **Trust Score più basso** in assoluto, identificati come anomalie dall'Isolation Forest.\n\n")
        
        for row in res.iter_rows(named=True):
            score = row['trust_score']
            user = row['user_id']
            item = row['parent_asin']
            rating = row.get('rating', 'N/A')
            title = row.get('title', 'N/A')
            text = row.get('review_text', row.get('text', ''))
            
            f.write(f"### Trust Score: `{score:.6f}` | Utente: `{user}` | Item: `{item}`\n")
            f.write(f"**Rating:** {rating} ⭐\n\n")
            f.write(f"**Titolo:** {title}\n\n")
            f.write(f"> {text}\n\n")
            f.write("---\n")
            
    print(f"File salvato con successo in {out_file}")

if __name__ == "__main__":
    extract_suspicious()
