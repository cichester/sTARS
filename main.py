"""
main.py — Trust-Aware Recommender System
=========================================
Orchestratore end-to-end della pipeline.
Esegue tutti gli step in sequenza, con caching intelligente
(se un artefatto intermedio esiste già, lo step viene skippato).

Uso:
    venv\\Scripts\\python.exe main.py              # Pipeline completa
    venv\\Scripts\\python.exe main.py --step ingest # Solo un singolo step
    venv\\Scripts\\python.exe main.py --help        # Aiuto
"""

import argparse
import logging
import os
import sys
import time
import pickle
import numpy as np
import polars as pl

from config import (
    BASE_DIR, DATA_DIR, RAW_DATA_PATH, PARQUET_5CORE_PATH,
    FEATURES_PATH, EMBEDDINGS_DIR, TEST_SIZE_RATIO, TOP_K_VALUES
)

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(BASE_DIR, "pipeline_run.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("main")

RANKINGS_DIR = DATA_DIR / "rankings"
RESULTS_DIR  = BASE_DIR / "results"
os.makedirs(RANKINGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,  exist_ok=True)


# ── Step 1: Ingestion ────────────────────────────────────────────────
def step_ingest():
    """Legge il JSONL raw (~21 GB), applica il filtro iterativo 5-core e salva il Parquet."""
    from src.ingestion import run_ingestion
    logger.info("═══ STEP 1/6: INGESTION & 5-CORE ═══")
    run_ingestion()
    df = pl.read_parquet(PARQUET_5CORE_PATH)
    logger.info(f"  ✓ Parquet 5-core: {df.height:,} righe, {df.width} colonne")


# ── Step 2: Preprocessing ───────────────────────────────────────────
def step_preprocess():
    """Applica la pulizia testuale (HTML, URL) e aggiunge la colonna review_text."""
    from src.preprocessing import clean_text_columns
    logger.info("═══ STEP 2/6: PREPROCESSING TESTO ═══")

    preprocessed_path = DATA_DIR / "electronics_5core_clean.parquet"
    if preprocessed_path.exists():
        logger.info(f"  → {preprocessed_path} esiste già, skip.")
        return

    df = pl.read_parquet(PARQUET_5CORE_PATH)
    df_clean = clean_text_columns(df)
    df_clean.write_parquet(preprocessed_path)
    logger.info(f"  ✓ Dataset pulito salvato: {df_clean.height:,} righe, colonna 'review_text' aggiunta")


# ── Step 3: Feature Engineering ──────────────────────────────────────
def step_features():
    """Estrae le 11 feature comportamentali per utente e salva il Parquet."""
    from src.feature_engineering import run_feature_engineering
    logger.info("═══ STEP 3/6: FEATURE ENGINEERING ═══")
    run_feature_engineering()
    features = pl.read_parquet(FEATURES_PATH)
    logger.info(f"  ✓ Feature comportamentali: {features.height:,} utenti, {features.width} colonne")


# ── Step 4: Embedding ───────────────────────────────────────────────
def step_embed():
    """Produce gli embedding Sentence-BERT e costruisce i profili User/Item."""
    from src.embedding_engine import run_embedding_pipeline
    logger.info("═══ STEP 4/6: EMBEDDING ENGINE (GPU) ═══")

    user_prof = EMBEDDINGS_DIR / "user_profiles.npy"
    item_prof = EMBEDDINGS_DIR / "item_profiles.npy"
    if user_prof.exists() and item_prof.exists():
        logger.info(f"  → Profili già presenti, skip.")
        return

    run_embedding_pipeline()
    logger.info("  ✓ Embedding e profili generati")


# ── Step 5: Anomaly Detection ───────────────────────────────────────
def step_anomaly():
    """Addestra Isolation Forest e calcola il trust_score per ogni utente."""
    from src.anomaly_detector import run_anomaly_detection
    logger.info("═══ STEP 5/6: ANOMALY DETECTION ═══")

    features = pl.read_parquet(FEATURES_PATH)
    if "trust_score" in features.columns:
        logger.info("  → trust_score già presente, skip.")
        return

    run_anomaly_detection()
    features = pl.read_parquet(FEATURES_PATH)
    logger.info(f"  ✓ Trust Score — min: {features['trust_score'].min():.4f}, max: {features['trust_score'].max():.4f}")


# ── Step 6: Fusion + Evaluation ─────────────────────────────────────
def step_evaluate():
    """
    Late fusion dei due stream, generazione ranking Baseline e Trust-Aware, 
    split temporale train/test e calcolo metriche (nDCG, Precision, Rank Shift).
    """
    from src.fusion import calculate_item_trust_factors, generate_ranking
    from src.evaluation import evaluate_recommender, calculate_rank_shift
    logger.info("═══ STEP 6/6: FUSION & EVALUATION ═══")

    # ── Caricamento ──
    df = pl.read_parquet(DATA_DIR / "electronics_5core_clean.parquet")
    features = pl.read_parquet(FEATURES_PATH)
    user_profiles = np.load(EMBEDDINGS_DIR / "user_profiles.npy", allow_pickle=True).item()
    item_profiles = np.load(EMBEDDINGS_DIR / "item_profiles.npy", allow_pickle=True).item()

    # ── Split temporale Train/Test ──
    logger.info("  Split temporale 80/20 basato su timestamp...")

    # Convertiamo timestamp se necessario
    if df.schema["timestamp"] in [pl.Int64, pl.Float64, pl.Int32]:
        df = df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"))
    else:
        df = df.with_columns(pl.col("timestamp").alias("datetime"))

    cutoff = df["datetime"].quantile(1.0 - TEST_SIZE_RATIO)
    train_df = df.filter(pl.col("datetime") <= cutoff)
    test_df  = df.filter(pl.col("datetime") > cutoff)
    logger.info(f"  Train: {train_df.height:,} — Test: {test_df.height:,}")

    # ── Ground truth per il test set: per ogni user, i parent_asin nel test ──
    test_ground_truth = {}
    for row in test_df.select(["user_id", "parent_asin"]).iter_rows():
        uid, pid = row
        test_ground_truth.setdefault(uid, set()).add(pid)

    # Filtriamo a utenti che compaiono sia nel train che nel test
    train_users = set(train_df["user_id"].unique().to_list())
    eval_users  = {u for u in test_ground_truth if u in train_users and u in user_profiles}
    test_ground_truth = {u: v for u, v in test_ground_truth.items() if u in eval_users}
    logger.info(f"  Utenti valutabili (presenti in train + test + profiles): {len(eval_users):,}")

    # ── Trust Factor per item ──
    item_trust_factors = calculate_item_trust_factors(train_df, features)

    # ── Profili filtrati su eval_users (per evitare matrici inutili) ──
    eval_user_profiles = {u: user_profiles[u] for u in eval_users}

    # ── Ranking ──
    max_k = max(TOP_K_VALUES)
    baseline_ranking = generate_ranking(eval_user_profiles, item_profiles, item_trust_factors, top_k=max_k, use_trust=False)
    trust_ranking    = generate_ranking(eval_user_profiles, item_profiles, item_trust_factors, top_k=max_k, use_trust=True)

    # ── Salvataggio ranking ──
    with open(RANKINGS_DIR / "baseline.pkl", "wb") as f:
        pickle.dump(baseline_ranking, f)
    with open(RANKINGS_DIR / "trust_aware.pkl", "wb") as f:
        pickle.dump(trust_ranking, f)

    # ── Metriche ──
    baseline_metrics = evaluate_recommender(baseline_ranking, test_ground_truth, k_list=TOP_K_VALUES)
    trust_metrics    = evaluate_recommender(trust_ranking,    test_ground_truth, k_list=TOP_K_VALUES)

    # ── Rank Shift ──
    rank_shifts = calculate_rank_shift(baseline_ranking, trust_ranking)
    avg_shift   = np.mean(list(rank_shifts.values())) if rank_shifts else 0.0

    # ── Report ──
    report_lines = [
        "# Evaluation Report — Trust-Aware Recommender System",
        "",
        f"**Data:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Train:** {train_df.height:,} review — **Test:** {test_df.height:,} review",
        f"**Utenti valutati:** {len(eval_users):,}",
        "",
        "## Confronto Metriche",
        "",
        "| Metrica | Baseline | Trust-Aware | Delta |",
        "|---------|----------|-------------|-------|",
    ]
    for met in sorted(baseline_metrics):
        b = baseline_metrics[met]
        t = trust_metrics[met]
        delta = t - b
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        report_lines.append(f"| {met} | {b:.4f} | {t:.4f} | {delta:+.4f} {arrow} |")

    report_lines += [
        "",
        "## Rank Shift",
        "",
        f"**Rank Shift medio (tutti gli item):** {avg_shift:+.2f}",
        "",
        "### Top 20 Item con maggior perdita di posizioni (sospetti manipolati)",
        "",
        "| Item | Avg Rank Shift |",
        "|------|----------------|",
    ]
    sorted_shifts = sorted(rank_shifts.items(), key=lambda x: x[1])[:20]
    for item, shift in sorted_shifts:
        report_lines.append(f"| {item} | {shift:+.2f} |")

    report_lines += [
        "",
        "> La flessione delle metriche classiche è una **feature**, non un bug.",
        "> Dimostra che il sistema penalizza attivamente contenuti spinti da review anomale.",
    ]

    report_path = RESULTS_DIR / f"evaluation_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"  ✓ Report salvato in {report_path}")

    # Stampa a console
    print("\n" + "═" * 60)
    print("  RISULTATI EVALUATION")
    print("═" * 60)
    print(f"  {'Metrica':<20} {'Baseline':>10} {'Trust-Aware':>12} {'Delta':>10}")
    print("  " + "─" * 54)
    for met in sorted(baseline_metrics):
        b = baseline_metrics[met]
        t = trust_metrics[met]
        delta = t - b
        print(f"  {met:<20} {b:>10.4f} {t:>12.4f} {delta:>+10.4f}")
    print(f"\n  Rank Shift medio: {avg_shift:+.2f}")
    print("═" * 60 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────
STEPS = {
    "ingest":     step_ingest,
    "preprocess":  step_preprocess,
    "features":   step_features,
    "embed":      step_embed,
    "anomaly":    step_anomaly,
    "evaluate":   step_evaluate,
}

def main():
    parser = argparse.ArgumentParser(
        description="Trust-Aware Recommender — Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--step", choices=list(STEPS.keys()),
        help="Esegui un singolo step:\n"
             "  ingest     → Ingestion & 5-core filtering\n"
             "  preprocess → Pulizia testo (HTML/URL)\n"
             "  features   → Feature Engineering comportamentale\n"
             "  embed      → Embedding Sentence-BERT (GPU)\n"
             "  anomaly    → Isolation Forest → Trust Score\n"
             "  evaluate   → Late Fusion + Metriche IR\n"
             "Se omesso, esegue tutti gli step in sequenza."
    )
    args = parser.parse_args()

    start = time.time()
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  Trust-Aware Recommender — Pipeline Runner          ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    if args.step:
        logger.info(f"Esecuzione singolo step: {args.step}")
        STEPS[args.step]()
    else:
        logger.info("Esecuzione PIPELINE COMPLETA (6 step)")
        for name, fn in STEPS.items():
            fn()

    elapsed = time.time() - start
    logger.info(f"Pipeline completata in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
