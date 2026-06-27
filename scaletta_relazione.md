# Scaletta della Relazione LaTeX — sTARS

> **Titolo completo:** sTARS: Spam-Trust-Aware Recommender System per la Rilevazione e Mitigazione delle Fake Review in Ambito E-commerce  
> **Corso:** Sistemi Intelligenti per Internet  
> **Dataset:** Amazon Review Data — Electronics (5-core)  
> **Ultimo aggiornamento:** 27 Giugno 2026

---

## Frontespizio

- Logo Università degli Studi Roma Tre
- Dipartimento di Ingegneria
- Corso di Laurea / Insegnamento: Sistemi Intelligenti per Internet
- Titolo del progetto
- Nome, cognome e matricola dello studente
- Anno Accademico
- Link alla repository GitHub

---

## Indice

- Indice dei contenuti (generato automaticamente da LaTeX con `\tableofcontents`)
- Elenco delle figure (`\listoffigures`)
- Elenco delle tabelle (`\listoftables`)

---

## 1. Introduzione

### 1.1 Contesto e Motivazione
- Il ruolo centrale delle recensioni online nelle decisioni d'acquisto
- Il fenomeno delle *fake review*: botnet, reviewer pagati, campagne coordinate
- Impatto economico e sulla fiducia degli utenti nelle piattaforme e-commerce
- Limiti dei filtri collaborativi e content-based tradizionali: ottimizzano su frequenza e coerenza delle interazioni, esattamente le metriche che gli spammer sanno simulare

### 1.2 Obiettivi del Progetto
- Identificare utenti con comportamento anomalo tramite feature comportamentali non supervisionate
- Generare un Trust Score continuo [0, 1] per ogni utente
- Combinare rilevanza semantica e fiducia in un'unica funzione di scoring (Late Fusion)
- Dimostrare che il sistema Trust-Aware produce un Rank Shift misurabile sui prodotti sospetti
- Validare la robustezza del sistema tramite attacchi avversariali simulati (Data Poisoning)

### 1.3 Panoramica dell'Architettura
- Paradigma a Late Fusion con tre moduli:
  - **Stream A** — Content-Based Engine (estrazione semantica via Transformer)
  - **Stream B** — Trust Scorer (Anomaly Detection comportamentale via Isolation Forest)
  - **Modulo C** — Late Fusion Ranking (combinazione dei due segnali)
- Schema architetturale ad alto livello (diagramma ASCII o TikZ)

### 1.4 Stack Tecnologico
- Tabella riassuntiva: Polars, Sentence-BERT, Isolation Forest (scikit-learn), PyTorch + CUDA, pytest
- Motivazione di ogni scelta tecnologica
- Hardware utilizzato: RTX 5070, 8 GB VRAM, CUDA 13.2

---

## 2. Ingegneria dei Dati e Preprocessing

### 2.1 Dataset di Riferimento
- Amazon Review Data (Electronics), variante 5-core
- Dimensioni originali: ~21 GB, ~44M record in formato JSONL
- Struttura dei record: `user_id`, `parent_asin`, `rating`, `text`, `title`, `timestamp`, `verified_purchase`, `helpful_vote`

### 2.2 Ingestion ed Efficientamento
- Utilizzo di Polars (lazy API) per la lettura streaming senza caricare tutto in RAM
- Filtro iterativo N-core (`MIN_INTERACTIONS = 10`): motivazione della scelta di 10 anziché 5 (sostenibilità computazionale su hardware consumer)
- Convergenza del filtro iterativo e dimensioni finali del dataset filtrato
- Serializzazione in formato Parquet (colonnare, compresso)

### 2.3 Preprocessing Testuale
- Rimozione tag HTML (BeautifulSoup) e URL (regex)
- Scelta di mantenere la struttura sintattica originale per non degradare il contesto NLP
- Nessuna rimozione di stopwords né stemming (preservazione del segnale semantico per Sentence-BERT)

### 2.4 Suddivisione Train / Test
- Split casuale con `TEST_SIZE_RATIO = 0.2`
- Dimensioni risultanti: ~4.9M training, ~1.2M test
- 231.535 utenti valutati nel run completo

---

## 3. Anomaly Detection Comportamentale (Stream B — Trust Scorer)

### 3.1 Il Problema della Ground Truth Assente
- In contesti reali, le etichette di "fake" o "genuine" non sono disponibili o sono esse stesse corrotte
- Motivazione dell'approccio non supervisionato: non dipendere da etichette potenzialmente inquinate
- Confronto con approcci supervisionati e semi-supervisionati (limiti)

### 3.2 Feature Comportamentali Estratte
- Tabella dettagliata delle 14 feature (11 comportamentali + 3 testuali):
  - `review_count`, `avg_rating`, `rating_std`, `rating_entropy`
  - `pct_extreme`, `pct_verified`, `avg_helpful`
  - `activity_span_days`, `burstiness`
  - `avg_text_length`, `unique_items_ratio`
  - `avg_ttr` (Type-Token Ratio), `avg_word_len`, `avg_punct_ratio`
- Spiegazione del segnale di anomalia per ciascuna feature
- Implementazione efficiente con Polars (aggregazioni native, nessun loop Python)

### 3.3 Isolation Forest
- Fondamenti teorici: isolamento geometrico degli outlier tramite partizionamento casuale (Liu et al., 2008)
- Formulazione matematica dell'Anomaly Score: $s(x_i, n) = 2^{-E[h(x_i)] / c(n)}$
- Iperparametri scelti: `n_estimators = 200`, `contamination = 0.05`, `random_state = 42`
- StandardScaler come preprocessing delle feature prima del fit
- Normalizzazione min-max dell'anomaly score per ottenere il Trust Score ∈ [0, 1]

### 3.4 Analisi della Distribuzione delle Anomalie
- Distribuzione dei Trust Score: istogramma + KDE con soglia di anomalia
  - **Figura:** `trust_score_distribution.png`
- Scatter Plot nello spazio delle feature (Burstiness vs Entropia), sdoppiato genuini/anomali
  - **Figura:** `behavioral_scatter.png`
- Discussione sulla multidimensionalità: utenti anomali a bassa burstiness ma con pattern anomali su altre feature (acquisti non verificati, entropia nulla, testi ripetitivi)
- Perché un modello ML è superiore a semplici filtri euristici

---

## 4. Motore di Raccomandazione Semantico (Stream A — Content-Based Engine)

### 4.1 Rappresentazioni Dense via Transformer
- Sentence-BERT (`all-MiniLM-L6-v2`): modello pre-trained su corpus NLI
- Output: embedding a 384 dimensioni per ogni recensione
- Inferenza GPU-accelerata con batch size = 1024

### 4.2 Profilo Utente Trust-Weighted
- Costruzione del vettore di profilo utente come media pesata degli embedding delle sue recensioni
- Il peso è il Trust Score dell'utente stesso
- Effetto: il profilo utente è robusto alla manipolazione fin dalla costruzione

### 4.3 Embedding dell'Item Trust-Weighted
- Embedding dell'item = media pesata degli embedding delle recensioni che lo riguardano
- Peso = Trust Score dell'autore di ciascuna recensione
- Effetto: gli item recensiti prevalentemente da bot hanno embedding meno influenzati dallo spam

### 4.4 Calcolo della Similarità
- Cosine similarity tra profilo utente e embedding di ciascun item candidato
- Ranking degli item per similarità decrescente

---

## 5. Late Fusion Ranking (Modulo C)

### 5.1 Funzione di Scoring Combinata
- Formula: $\text{Score}_{u,i} = \text{Sim}(u, i) \times \text{TrustFactor}(i)$
- $\text{TrustFactor}(i)$ = media dei Trust Score degli utenti che hanno recensito l'item $i$
- Interpretazione: un item recensito prevalentemente da spammer viene penalizzato indipendentemente dalla rilevanza semantica

### 5.2 Generazione del Ranking Finale
- Ordinamento degli item per Score decrescente
- Selezione delle Top-K raccomandazioni ($K \in \{5, 10, 20\}$)
- Confronto con il ranking Baseline (senza il fattore Trust)

---

## 6. Valutazione Quantitativa

### 6.1 Metriche di Information Retrieval
- **Precision@K**: frazione di item rilevanti nelle prime K raccomandazioni
- **nDCG@K**: qualità dell'ordinamento con penalizzazione logaritmica per posizione
- **Rank Shift**: variazione della posizione media di un item tra Baseline e Trust-Aware

### 6.2 Risultati Sperimentali
- Tabella comparativa Baseline vs Trust-Aware (dataset completo, 5 Giugno 2026)
- Miglioramenti osservati: +29% — +58% su tutte le metriche
- **Figura:** `recommendation_metrics.png` — Grafico a barre del confronto
- Rank Shift medio: −7.16 posizioni
- Tabella dei Top 20 item con maggior perdita di posizioni (sospetti manipolati)

### 6.3 Interpretazione dei Risultati
- Perché il Trust-Aware migliora le metriche anziché degradarle (controintuitivo rispetto all'ipotesi iniziale)
- Effetto della pulizia del rumore sui profili utente e sugli embedding degli item
- Il Rank Shift negativo come conferma del filtraggio attivo

### 6.4 Validazione Indiretta dell'Anomaly Detection
- Argomento: se il sistema classificasse erroneamente utenti genuini come spammer, le metriche peggiorerebbero
- Il miglioramento delle metriche è una prova indiretta che i profili rimossi erano effettivamente rumorosi
- Analisi qualitativa manuale delle recensioni con Trust Score più basso (file `suspicious_reviews.md`)

---

## 7. Valutazione Avversariale (Data Poisoning)

### 7.1 Modello di Minaccia
- Obiettivo dell'attaccante: promuovere un Target Item nel ranking tramite l'iniezione di bot sintetici
- Tre tipologie di attacco implementate:
  - **Average Attack**: bot assegnano rating vicini alla media globale per mimetizzarsi
  - **Bandwagon Attack**: bot recensiscono i prodotti più popolari per massimizzare l'impatto
  - **GenAI Attack**: bot generano testi plausibili tramite LLM per eludere filtri testuali

### 7.2 Protocollo Sperimentale
- Intensità crescente: 50, 100, 200 bot iniettati
- Target item scelto casualmente per ogni run
- Ricalcolo completo della pipeline (feature, anomaly detection, embedding, ranking)
- Metriche misurate: Bot Detection Rate, Posizione del Target, Hit Rate@20

### 7.3 Risultati sotto Attacco
- Tabelle comparative per modello SBERT e RoBERTa
- Il sistema Trust-Aware mantiene il Target Item fuori dal ranking (posizione 50) mentre la Baseline è vulnerabile (fino a posizione 17)
- **Figura:** `adversarial_resilience.png` — Andamento del rank del target al crescere dei bot
- Discussione sulla Bot Detection Rate (0% formale ma neutralizzazione effettiva via Trust Score)

### 7.4 Discussione sulla Robustezza
- Il Trust-Aware non ha bisogno di "rilevare" esplicitamente i bot per neutralizzarli
- Il meccanismo di pesatura per Trust Score attenua automaticamente il contributo degli spammer
- Limiti: attacchi molto sofisticati con profili a lungo termine potrebbero eludere il sistema

---

## 8. Sviluppi Futuri

### 8.1 Fine-Tuning Task-Specific con RoBERTa + LoRA
- Motivazione: modello general-purpose vs. modello specializzato sul dominio Electronics
- Architettura: RoBERTa-base + PEFT/LoRA (rank 16, alpha 32)
- Training su triple contrastive (anchor, positive, negative) estratte dalle interazioni utente-item
- 4-bit quantization via bitsandbytes per il fine-tuning su 8 GB VRAM

### 8.2 Possibili Estensioni
- Integrazione di segnali temporali (drift detection, evoluzione del trust nel tempo)
- Graph Neural Networks per modellare relazioni utente-item-utente
- Explainability: report automatico per l'utente finale sul "perché" di una raccomandazione

---

## 9. Conclusioni
- Riepilogo dei contributi principali di sTARS
- Il layer di Anomaly Detection comportamentale migliora sia la robustezza che l'accuratezza
- Risultati: +40–70% su nDCG e Precision, Rank Shift medio −7 posizioni, resilienza totale al Data Poisoning
- Pipeline modulare, cacheabile e scalabile su hardware consumer

---

## Bibliografia
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. ICDM.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
- Ni, J., Li, J., & McAuley, J. (2019). *Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects*. EMNLP. (Amazon Review Dataset)
- Luca, M. (2016). *Reviews, Reputation, and Revenue: The Case of Yelp.com*. HBS Working Paper.
- Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM.

---

## Appendice (Opzionale)
- Struttura completa del repository
- Snippet di codice rilevanti (feature engineering, Isolation Forest, Late Fusion)
- Log di esecuzione della pipeline completa
- Tabelle complete degli adversarial report

---

## Figure da includere

| Figura | File | Sezione |
|--------|------|---------|
| Distribuzione dei Trust Score | `results/plots/trust_score_distribution.png` | §3.4 |
| Scatter Plot Genuini vs Anomali | `results/plots/behavioral_scatter.png` | §3.4 |
| Confronto Metriche Baseline vs Trust-Aware | `results/plots/recommendation_metrics.png` | §6.2 |
| Resilienza Avversariale | `results/plots/adversarial_resilience.png` | §7.3 |
| Diagramma Architetturale (da creare in TikZ) | — | §1.3 |
