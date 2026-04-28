# Project Ground Truth: Trust-Aware Recommender System

## Obiettivo
Sviluppare un sistema di raccomandazione che integra l'Anomaly Detection per mitigare l'impatto delle Fake Reviews.

## Stack Tecnico
- **Dataset:** Amazon Review Data (Electronics, 5-core).
- **Processing:** Polars (per gestione memoria efficiente) o Pandas a chunk.
- **NLP:** Hugging Face Transformers (RoBERTa-base) con PEFT/LoRA.
- **Anomaly Detection:** Scikit-learn (Isolation Forest).
- **Hardware Target:** 32GB RAM, 8GB VRAM (RTX 5070).

## Architettura (Late Fusion)
1. **Stream A (Content):** Embedding semantici del testo (Transformer).
2. **Stream B (Behavioral):** Analisi non supervisionata delle anomalie (Isolation Forest su timestamp, rating, frequenza).
3. **Fusion Logic:** Score Finale = (Similarity * Trust_Factor).

## Vincoli di Progetto
- Nessuna rimozione di stop-words per i modelli Transformer.
- Pre-processing minimo (pulizia HTML/URL).
- Valutazione tramite nDCG e Rank Shift.

## Schema Dataset (Electronics.jsonl)
Ogni record contiene:
| Campo             | Tipo   | Descrizione                        |
|-------------------|--------|------------------------------------|
| rating            | float  | Punteggio 1.0-5.0                  |
| title             | str    | Titolo della recensione            |
| text              | str    | Corpo della recensione             |
| images            | list   | Immagini allegate                  |
| asin              | str    | ID prodotto (variante)             |
| parent_asin       | str    | ID prodotto (parent)               |
| user_id           | str    | ID utente anonimizzato             |
| timestamp         | int    | Epoch milliseconds                 |
| helpful_vote      | int    | Conteggio voti utili               |
| verified_purchase | bool   | Flag acquisto verificato           |

## Note Operative
- Il dataset corrente è la versione FULL (~44M record, ~21GB). Il filtraggio 5-core va applicato in fase di Ingestion.
- Il filtraggio 5-core richiede: ogni user_id e ogni parent_asin devono avere almeno 5 interazioni.
