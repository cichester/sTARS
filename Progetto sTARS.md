# **Documento di Progetto: Trust-Aware Recommender System**

**Un Approccio Ibrido per la Rilevazione e Mitigazione delle Fake Review in Ambito E-commerce**

## **1\. Contesto e Motivazione**

I Sistemi di Raccomandazione (RS) tradizionali assumono che le interazioni utente-item (rating, click) siano espressione genuina delle preferenze. Tuttavia, in domini altamente competitivi come l'e-commerce, l'ecosistema è inquinato da *Fake Reviews* generate artificialmente per manipolare il ranking dei prodotti. L'obiettivo di questo progetto è superare i limiti dei classici filtri collaborativi o basati sul contenuto, architettando un **Trust-Aware Recommender System**. Il sistema non si limiterà a calcolare la similarità tra profilo utente e item, ma introdurrà un layer di *Anomaly Detection* per penalizzare i contenuti spinti da comportamenti non genuini.

## **2\. Dataset e Preprocessing**

Il progetto utilizzerà l'**Amazon Review Data (Electronics, 5-core)**. La scelta della variante *5-core* (utenti e item con almeno 5 interazioni) è una decisione architetturale mirata a mitigare il problema della *sparsity* e garantire un volume minimo di dati storici per profilare correttamente il comportamento di ciascun utente nel modulo di anomaly detection. Il preprocessing testuale manterrà la struttura sintattica originale per non degradare il contesto necessario ai modelli di *Natural Language Processing* (NLP).

## **3\. Architettura del Sistema**

Il sistema seguirà un paradigma a *Late Fusion*, scomposto in tre moduli principali:

* **Modulo A: Estrazione Semantica (Content-Based Engine):** Utilizzo di modelli basati su architettura Transformer (es. Sentence-BERT)  per mappare il testo delle recensioni in *dense vector embeddings*. Questo modulo valuterà la pura rilevanza semantica tra gli interessi dell'utente e il prodotto.

* **Modulo B: Behavioral Anomaly Detection (Trust Scorer):** Un approccio non supervisionato mirato a estrarre feature comportamentali dai log (es. *Burstiness* delle recensioni, deviazione dal rating medio, estremizzazione dei voti). Un algoritmo come la *Isolation Forest* genererà un *Anomaly Score* per isolare profili assimilabili a spammer o botnet, aggirando il problema delle etichette (Ground Truth) potenzialmente inquinate.  
* **Modulo C: Late Fusion Ranking:** La funzione di utilità finale combinerà il punteggio di similarità (Modulo A) con una penalità proporzionale all'Anomaly Score (Modulo B), filtrando di fatto i prodotti spinti da campagne di manipolazione.

## **4\. Metodologia di Valutazione (Fase 1-3)**

Inizialmente, la valutazione seguirà i crismi dell'Information Retrieval classico. Verrà calcolata una baseline utilizzando solo il Modulo A, misurando le metriche classiche come **nDCG (Normalized Discounted Cumulative Gain)** e **Precision@K**. Nell'integrare il Modulo C (Trust-Aware), si prevede e si documenterà una fisiologica *flessione* di tali metriche classiche. Tale fenomeno non indicherà un degrado algoritmico, ma dimostrerà che il sistema sta attivamente nascondendo prodotti storicamente molto cliccati/votati, ma identificati come manipolati.

## **5\. Obiettivo Futuro: Adversarial Evaluation (Fase 4\)**

Per validare empiricamente la robustezza del sistema e dimostrare inequivocabilmente che la flessione delle metriche tradizionali è una "feature" e non un "bug", lo sviluppo futuro prevede un test in ambiente avversariale (*Data Poisoning*).

Verranno iniettati nel dataset bot sintetici programmati per eseguire attacchi noti (es. *Average Attack* o *Bandwagon Attack*) per spingere prodotti civetta ("Target").

La validazione finale misurerà il **Rank Shift**: il successo del sistema sarà determinato dalla sua capacità di assorbire l'attacco, identificare i bot e far precipitare il prodotto Target fuori dalla lista delle raccomandazioni (Top-K), dimostrando una resilienza superiore rispetto ai RS classici.

