import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurazione dello stile grafico per renderlo moderno e premium
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

# Palette colori curata
COLORS = {
    'baseline': '#9E9E9E',      # Grigio elegante per la baseline
    'trust_aware': '#4361EE',    # Blu reale per il sistema Trust-Aware
    'anomaly': '#F72585',        # Rosa/Fucsia vibrante per gli spammer
    'genuine': '#4CC9F0',        # Celeste per gli utenti genuini
    'grid': '#E5E5E5'
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Assicura che la cartella dei grafici esista
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_recommendation_metrics():
    """1. Confronto delle metriche di raccomandazione (Baseline vs Trust-Aware)"""
    print("Generazione Grafico 1: Metriche di Raccomandazione...")
    
    # Dati dal dataset completo (Sezione 7.2 della Relazione)
    metrics = ['Precision@5', 'Precision@10', 'Precision@20', 'nDCG@5', 'nDCG@10', 'nDCG@20']
    baseline_vals = [0.0033, 0.0029, 0.0024, 0.0041, 0.0048, 0.0063]
    trust_aware_vals = [0.0050, 0.0041, 0.0032, 0.0065, 0.0072, 0.0088]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Content-Based SBERT)', color=COLORS['baseline'], edgecolor='none', alpha=0.85)
    rects2 = ax.bar(x + width/2, trust_aware_vals, width, label='Trust-Aware (sTARS)', color=COLORS['trust_aware'], edgecolor='none', alpha=0.95)
    
    ax.set_ylabel('Valore Metrica')
    ax.set_title('Confronto delle Performance di Raccomandazione (Dataset Completo 5-core)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Aggiunta delle percentuali di miglioramento sopra le barre
    for i in range(len(metrics)):
        improvement = ((trust_aware_vals[i] - baseline_vals[i]) / baseline_vals[i]) * 100
        val = trust_aware_vals[i]
        ax.annotate(f'+{improvement:.1f}%',
                    xy=(i + width/2, val),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['trust_aware'])
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.grid(axis='y', linestyle='--', alpha=0.5, color=COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "recommendation_metrics.png", dpi=300)
    plt.close()

def plot_trust_score_distribution():
    """2. Distribuzione dei Trust Score"""
    print("Generazione Grafico 2: Distribuzione Trust Score...")
    features_path = DATA_DIR / "features_behavioral.parquet"
    if not features_path.exists():
        print(f"File {features_path} non trovato. Impossibile generare la distribuzione dei trust score.")
        return
        
    df = pl.read_parquet(features_path)
    trust_scores = df["trust_score"].to_list()
    
    plt.subplots(figsize=(9, 5.5))
    
    # Distribuzione (KDE + Hist)
    sns.histplot(trust_scores, bins=40, kde=True, color=COLORS['trust_aware'], edgecolor='white', alpha=0.7)
    
    # Linea di soglia teorica al 5% (contamination rate)
    threshold = np.percentile(trust_scores, 5)
    plt.axvline(threshold, color=COLORS['anomaly'], linestyle='--', linewidth=2, 
                label=f'Soglia Anomalia (5% Contamination): {threshold:.3f}')
    
    plt.title('Distribuzione dei Trust Score calcolati da Isolation Forest')
    plt.xlabel('Trust Score (0.0 = Spammer, 1.0 = Genuino)')
    plt.ylabel('Frequenza (Numero Utenti)')
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "trust_score_distribution.png", dpi=300)
    plt.close()

def plot_behavioral_scatter():
    """3. Scatter Plot delle Feature Comportamentali (Burstiness vs Entropia)"""
    print("Generazione Grafico 3: Scatter Plot delle Feature...")
    features_path = DATA_DIR / "features_behavioral.parquet"
    if not features_path.exists():
        print("File non trovato. Salto scatter plot.")
        return
        
    df = pl.read_parquet(features_path)
    
    # Aggiungiamo etichetta binaria basata sul 5% dei punteggi più bassi (le anomalie dell'IF)
    threshold = np.percentile(df["trust_score"].to_numpy(), 5)
    df = df.with_columns(
        pl.when(pl.col("trust_score") <= threshold).then(pl.lit("Sospetto (Spammer)")).otherwise(pl.lit("Genuino")).alias("status")
    )
    
    # Campioniamo per evitare sovraffollamento nel plot se il dataset è molto grande
    sample_size = min(df.height, 10000)
    df_sample = df.sample(n=sample_size, seed=42)
    
    # Per essere sicuri di visualizzare le anomalie anche nel campione, 
    # forziamo l'inclusione di una quota di anomalie
    anomalies = df.filter(pl.col("status") == "Sospetto (Spammer)")
    genuines = df.filter(pl.col("status") == "Genuino")
    
    n_anom_show = min(anomalies.height, 1000)
    n_gen_show = sample_size - n_anom_show
    
    df_plot = pl.concat([
        genuines.sample(n=n_gen_show, seed=42),
        anomalies.sample(n=n_anom_show, seed=42)
    ])
    
    pandas_df = df_plot.to_pandas()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    
    # Plot degli utenti genuini (Sinistra)
    sns.scatterplot(
        data=pandas_df[pandas_df['status'] == 'Genuino'],
        x='burstiness', y='rating_entropy',
        color=COLORS['genuine'], alpha=0.4, s=15, legend=False,
        ax=ax1
    )
    ax1.set_title('Utenti Genuini', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Burstiness (Max review in 24h)')
    ax1.set_ylabel('Rating Entropy (Entropia di Shannon)')
    ax1.grid(True, linestyle='--', alpha=0.5, color=COLORS['grid'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot delle anomalie (Destra)
    sns.scatterplot(
        data=pandas_df[pandas_df['status'] == 'Sospetto (Spammer)'],
        x='burstiness', y='rating_entropy',
        color=COLORS['anomaly'], alpha=0.8, s=40, edgecolor='black', linewidth=0.5, legend=False,
        ax=ax2
    )
    ax2.set_title('Utenti Sospetti (Anomalie)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Burstiness (Max review in 24h)')
    ax2.grid(True, linestyle='--', alpha=0.5, color=COLORS['grid'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle('Spazio delle Feature: Burstiness vs Entropia del Rating (Confronto)', fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "behavioral_scatter.png", dpi=300)
    plt.close()

def plot_adversarial_resilience():
    """4. Grafico di Resilienza Avversariale (RoBERTa Average Attack)"""
    print("Generazione Grafico 4: Resilienza Avversariale...")
    
    # Dati da results/adversarial_report_20260606_110023.md (Modello RoBERTa Average Attack)
    num_bots = [0, 50, 100, 200]
    
    # Rank del Target Item (1 è il primo consigliato, 50 è l'ultimo o fuori classifica)
    # Più il valore è BASSO (vicino a 1), più l'attacco ha avuto successo.
    # Più il valore è ALTO (vicino a 50), più il sistema ha respinto l'attacco.
    baseline_rank = [50.00, 24.00, 23.38, 17.00]
    trust_aware_rank = [50.00, 50.00, 50.00, 50.00]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    ax.plot(num_bots, baseline_rank, marker='o', linewidth=2.5, color=COLORS['baseline'], 
            label='Baseline (RoBERTa Content-Based)', markersize=8)
    ax.plot(num_bots, trust_aware_rank, marker='s', linewidth=2.5, color=COLORS['trust_aware'], 
            label='Trust-Aware (sTARS)', markersize=8)
    
    # Poiché 1 è il rank migliore (Top-1) e 50 è il peggiore (fuori top-50), invertiamo l'asse Y
    # così che "salire nel ranking" (cioè il successo del bot di posizionare il target in alto) 
    # sia rappresentato graficamente da una curva che sale.
    ax.set_ylim(52, 10)  # Invertito da 52 a 10 per mostrare la salita del ranking
    
    # Etichette asse Y specifiche
    ax.set_yticks([50, 40, 30, 20, 10])
    ax.set_yticklabels(['50 (Fuori Top-K)', '40', '30', '20', '10 (Top-10)'])
    
    ax.set_title('Resilienza all\'Attacco Avversariale (Average Attack su RoBERTa)')
    ax.set_xlabel('Numero di Bot Iniettati (Intensità dell\'Attacco)')
    ax.set_ylabel('Posizione Media del Target Item (Rank)')
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Evidenziazione grafica del successo del sistema
    ax.fill_between(num_bots, baseline_rank, trust_aware_rank, color='#4361EE', alpha=0.1, 
                    label='Zona di Mitigazione dell\'Attacco')
    
    # Aggiunta di note di testo sul grafico
    ax.text(120, 48, 'Attacco Mitigato (Target Represso)', color=COLORS['trust_aware'], fontweight='bold')
    ax.text(120, 26, 'Attacco Riuscito (Target Promosso)', color='#777777', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "adversarial_resilience.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Inizio generazione grafici per la relazione di sTARS...")
    plot_recommendation_metrics()
    plot_trust_score_distribution()
    plot_behavioral_scatter()
    plot_adversarial_resilience()
    print("Grafici generati con successo nella cartella: results/plots/")
