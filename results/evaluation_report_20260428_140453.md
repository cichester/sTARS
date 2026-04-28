# Evaluation Report — Trust-Aware Recommender System

**Data:** 2026-04-28 14:04
**Train:** 4,939,178 review — **Test:** 1,234,794 review
**Utenti valutati:** 231,535

## Confronto Metriche

| Metrica | Baseline | Trust-Aware | Delta |
|---------|----------|-------------|-------|
| Precision@10 | 0.0029 | 0.0041 | +0.0012 ↑ |
| Precision@20 | 0.0024 | 0.0032 | +0.0007 ↑ |
| Precision@5 | 0.0033 | 0.0050 | +0.0017 ↑ |
| nDCG@10 | 0.0048 | 0.0072 | +0.0024 ↑ |
| nDCG@20 | 0.0063 | 0.0088 | +0.0025 ↑ |
| nDCG@5 | 0.0041 | 0.0065 | +0.0024 ↑ |

## Rank Shift

**Rank Shift medio (tutti gli item):** -7.16

### Top 20 Item con maggior perdita di posizioni (sospetti manipolati)

| Item | Avg Rank Shift |
|------|----------------|
| B004I1JDH8 | -20.00 |
| B00LIS5KD0 | -20.00 |
| B0BQRVK3BG | -20.00 |
| B005K8AUOI | -20.00 |
| B07CZ12WKL | -20.00 |
| B0002XMZOO | -20.00 |
| B016W4IAR2 | -20.00 |
| B00284ALEG | -20.00 |
| B004G8QO5C | -20.00 |
| B01E09S7BU | -20.00 |
| B075SFLWKX | -20.00 |
| B076X3K4N3 | -20.00 |
| B07S4GK8DD | -20.00 |
| B07TRKYZCH | -20.00 |
| B0091JRPJK | -20.00 |
| B08FS8D92X | -20.00 |
| B005S0KE7G | -20.00 |
| B002WSPHBA | -20.00 |
| B004QITD5U | -20.00 |
| B07KNDRBRV | -20.00 |

> La flessione delle metriche classiche è una **feature**, non un bug.
> Dimostra che il sistema penalizza attivamente contenuti spinti da review anomale.