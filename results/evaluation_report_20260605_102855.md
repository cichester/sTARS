# Evaluation Report — Trust-Aware Recommender System

**Data:** 2026-06-05 10:28
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
| B004BRMV6O | -20.00 |
| B01BSNPKZM | -20.00 |
| B009WRN6MQ | -20.00 |
| B086S7DK1H | -20.00 |
| B07TF38PXT | -20.00 |
| B0C557416V | -20.00 |
| B004YW7W1A | -20.00 |
| B07K3ZLMP7 | -20.00 |
| B00DS5ZMFW | -20.00 |
| B00606TOX2 | -20.00 |
| B079BBK5B7 | -20.00 |
| B000S5BFS8 | -20.00 |
| B016UJMADE | -20.00 |
| B00BFR77SK | -20.00 |
| B07CLZB87W | -20.00 |
| B07VDF5VW5 | -20.00 |
| B003MXCXN4 | -20.00 |
| B0054ENBNK | -20.00 |
| B00AZS8PMM | -20.00 |
| B07RQTT79T | -20.00 |

> La flessione delle metriche classiche è una **feature**, non un bug.
> Dimostra che il sistema penalizza attivamente contenuti spinti da review anomale.