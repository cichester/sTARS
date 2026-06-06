# Adversarial Evaluation Report

**Data:** 2026-06-05 15:39

Questo report confronta la resilienza del sistema Baseline rispetto al Trust-Aware sotto attacchi di Data Poisoning a intensità progressiva.

## Modello: SBERT

### 1. Average Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B081359HVX` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B000UCJ874` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B09M8BSPTK` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

### 2. Bandwagon Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B00414ROMS` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B09FQ1T999` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B07B45FWQY` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

---

## Modello: RoBERTa

### 1. Average Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B09LDDNK2V` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B0BLKQKDNN` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B0BKG391FV` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

### 2. Bandwagon Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B002M3SO10` | 0.00% | 22.93 | 50.00 | 0.35% | 0.00% |
| 100 | `B086W3L2XB` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B08L33R3V1` | 0.00% | 4.00 | 50.00 | 0.05% | 0.00% |

---

> **Conclusioni:** Se la posizione media nel Trust-Aware è vicina a 50 (fuori ranking) e l'Hit Rate è prossimo allo 0%, mentre nella Baseline sono alti, significa che l'Anomaly Detection ha neutralizzato con successo l'attacco, dimostrando la robustezza del sistema.
