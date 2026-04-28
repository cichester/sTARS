# Evaluation Report — Trust-Aware Recommender System

**Data:** 2026-04-16 15:38
**Train:** 238,647 review — **Test:** 59,662 review
**Utenti valutati:** 8,067

## Confronto Metriche

| Metrica | Baseline | Trust-Aware | Delta |
|---------|----------|-------------|-------|
| Precision@10 | 0.0136 | 0.0203 | +0.0067 ↑ |
| Precision@20 | 0.0116 | 0.0164 | +0.0048 ↑ |
| Precision@5 | 0.0151 | 0.0243 | +0.0092 ↑ |
| nDCG@10 | 0.0178 | 0.0294 | +0.0116 ↑ |
| nDCG@20 | 0.0218 | 0.0348 | +0.0130 ↑ |
| nDCG@5 | 0.0165 | 0.0282 | +0.0117 ↑ |

## Rank Shift

**Rank Shift medio (tutti gli item):** -7.29

### Top 20 Item con maggior perdita di posizioni (sospetti manipolati)

| Item | Avg Rank Shift |
|------|----------------|
| B009JPBPWO | -20.00 |
| B0C682GZ5X | -20.00 |
| B00WUIB22U | -20.00 |
| B0198HIF7K | -20.00 |
| B00EOI3VC8 | -20.00 |
| B00MY05GNA | -19.00 |
| B001OCY3RY | -19.00 |
| B00DVRUTXW | -19.00 |
| B07N2ZYKSL | -19.00 |
| B0C678Z6KG | -19.00 |
| B002J9HBIO | -19.00 |
| B09NCBBB7P | -19.00 |
| B092CW6CWC | -19.00 |
| B003Z4G3I6 | -19.00 |
| B014R4IFO2 | -18.33 |
| B003VAK1I2 | -18.00 |
| B007PJ4Q4A | -18.00 |
| B00M14VAD4 | -18.00 |
| B07WLT77K8 | -18.00 |
| B0BWMXQZ32 | -18.00 |

> La flessione delle metriche classiche è una **feature**, non un bug.
> Dimostra che il sistema penalizza attivamente contenuti spinti da review anomale.