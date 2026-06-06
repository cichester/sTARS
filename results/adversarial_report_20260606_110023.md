# Adversarial Evaluation Report

**Data:** 2026-06-06 11:00

Questo report confronta la resilienza del sistema Baseline rispetto al Trust-Aware sotto attacchi di Data Poisoning a intensità progressiva.

## Modello: SBERT

### 1. Average Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B08KG29PKS` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B00XII32T2` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B00MMN59KY` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

### 2. Bandwagon Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B09MRNM214` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B07W5KCMP9` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B00AQUGNEG` | 0.00% | 17.20 | 50.00 | 0.10% | 0.00% |

### 3. GenAI Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B0083K9I8Y` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B09NVV747N` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B0BD8613BP` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

---

## Modello: RoBERTa

### 1. Average Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B00000K3RG` | 0.00% | 24.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B00T8H16UE` | 0.00% | 23.38 | 50.00 | 0.15% | 0.00% |
| 200 | `B00WQYCKSS` | 0.00% | 17.00 | 50.00 | 0.10% | 0.00% |

### 2. Bandwagon Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B08TV3742G` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B0BCKF25Q2` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B09257J3TW` | 0.00% | 45.00 | 50.00 | 0.00% | 0.00% |

### 3. GenAI Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B0713QRWB3` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B0BLSSZ62J` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B07GPGTPGM` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

---

> **Conclusioni:** Se la posizione media nel Trust-Aware è vicina a 50 (fuori ranking) e l'Hit Rate è prossimo allo 0%, mentre nella Baseline sono alti, significa che l'Anomaly Detection ha neutralizzato con successo l'attacco, dimostrando la robustezza del sistema.
