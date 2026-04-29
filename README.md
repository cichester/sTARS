# sTARS: Trust-Aware Recommender System

Un approccio ibrido per la rilevazione e mitigazione delle *Fake Review* in ambito E-commerce.

## Cosa fa il sistema

I Sistemi di Raccomandazione (RS) tradizionali assumono che le interazioni utente-item siano espressione genuina delle preferenze. Tuttavia, l'ecosistema dell'e-commerce è spesso inquinato da recensioni generate artificialmente per manipolare il ranking dei prodotti. 

**sTARS** supera i limiti dei classici filtri introducendo un layer di *Anomaly Detection* per penalizzare i contenuti spinti da comportamenti non genuini. L'architettura del sistema si basa su tre moduli principali (paradigma a Late Fusion):
1. **Estrazione Semantica (Content-Based Engine):** Utilizza modelli basati su Transformer (come Sentence-BERT) per mappare il testo delle recensioni in *dense vector embeddings*, valutando la rilevanza semantica pura tra utente e prodotto.
2. **Behavioral Anomaly Detection (Trust Scorer):** Analizza le feature comportamentali (es. *burstiness* delle recensioni, deviazione dal rating medio) e utilizza *Isolation Forest* in modo non supervisionato per generare un *Anomaly Score* e isolare profili assimilabili a spammer o botnet.
3. **Late Fusion Ranking:** Combina il punteggio di similarità (Modulo A) con una penalità proporzionale all'Anomaly Score (Modulo B), filtrando attivamente i prodotti spinti da campagne di manipolazione. La flessione delle metriche classiche di IR (come nDCG) nella valutazione finale è una *feature* ricercata, in quanto indica che il sistema sta attivamente "nascondendo" prodotti manipolati ma storicamente molto votati.

## Dataset

Il progetto utilizza il dataset **Amazon Review Data (Electronics)**. Il sistema applica automaticamente un filtro per estrarre il nucleo *5-core* (utenti e item con almeno 5 interazioni) per mitigare il problema della *sparsity*.

🔗 **Download Dataset:** [Electronics.jsonl.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz)

*(Assicurati di scaricare il dataset e posizionarlo nella directory `data/` o nel percorso `RAW_DATA_PATH` definito in `config.py` prima di lanciare la pipeline).*

## Come farlo partire

Il progetto è gestito da uno script di orchestrazione end-to-end (`main.py`). Il sistema implementa un meccanismo di caching intelligente: se un artefatto intermedio esiste già (es. file parquet preprocessati o numpy array degli embedding), quello step viene saltato per risparmiare tempo nelle esecuzioni successive.

1. **Pre-requisiti ed Environment:**
   Assicurati di avere le dipendenze installate. Attiva l'ambiente virtuale:
   ```bash
   venv\Scripts\activate
   ```
   *(Oppure tramite il path diretto: `venv\Scripts\python.exe main.py`)*

2. **Esecuzione completa della Pipeline:**
   Per lanciare l'intera pipeline in sequenza (Ingestion, Preprocessing, Feature Engineering, Embedding, Anomaly Detection e Evaluation):
   ```bash
   python main.py
   ```

3. **Esecuzione di step specifici:**
   È possibile eseguire i singoli step della pipeline passando l'argomento `--step`:
   ```bash
   python main.py --step ingest      # 1. Lettura JSONL, filtro 5-core e salvataggio Parquet
   python main.py --step preprocess  # 2. Pulizia del testo (HTML, URL)
   python main.py --step features    # 3. Estrazione feature comportamentali
   python main.py --step embed       # 4. Generazione profili ed Embedding Sentence-BERT
   python main.py --step anomaly     # 5. Addestramento Isolation Forest e calcolo Trust Score
   python main.py --step evaluate    # 6. Late Fusion, ranking Train/Test e calcolo metriche
   ```

4. **Aiuto CLI:**
   ```bash
   python main.py --help
   ```

## Output e Log

- I risultati finali, inclusi i report delle metriche e del *Rank Shift*, vengono salvati in formato Markdown all'interno della cartella `results/`.
- I ranking intermedi e le baseline serializzate si trovano in `data/rankings/`.
- L'intera esecuzione viene tracciata e i log sono salvati nel file `pipeline_run.log`.
