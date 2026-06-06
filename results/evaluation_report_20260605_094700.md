# Evaluation Report — Trust-Aware Recommender System

**Data:** 2026-06-05 09:47
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
| B07PJVKX8J | -20.00 |
| B0C231T5HV | -20.00 |
| B0B7P5GHH4 | -20.00 |
| B001P05NJC | -20.00 |
| B00X7XFJSK | -20.00 |
| B085FSQFCY | -20.00 |
| B005STXPOG | -20.00 |
| B088H5YVYF | -20.00 |
| B0755B945G | -20.00 |
| B004YW7W1A | -20.00 |
| B01LXFBERB | -20.00 |
| B016W4IAR2 | -20.00 |
| B073PN5LYL | -20.00 |
| B00CR8L26E | -20.00 |
| B01M24PNI3 | -20.00 |
| B087X1K4L7 | -20.00 |
| B07MVP819S | -20.00 |
| B0B37YMFK2 | -20.00 |
| B07JKSD1MK | -20.00 |
| B07SDGRQGB | -20.00 |

> La flessione delle metriche classiche è una **feature**, non un bug.
> Dimostra che il sistema penalizza attivamente contenuti spinti da review anomale.