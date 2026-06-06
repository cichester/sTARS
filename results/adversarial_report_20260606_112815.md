# Adversarial Evaluation Report

**Data:** 2026-06-06 11:28

Questo report confronta la resilienza del sistema Baseline rispetto al Trust-Aware sotto attacchi di Data Poisoning a intensità progressiva.

## Modello: SBERT

### 1. Average Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B0BVM5DFD4` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B00BGETYPG` | 0.00% | 42.67 | 50.00 | 0.00% | 0.00% |
| 200 | `B088LND513` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

### 2. Bandwagon Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B07YHK95Z6` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B004D39CI6` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B00E68LFC4` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

### 3. GenAI Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B00C7N3WN0` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B011TVK75Q` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B0018MMP5C` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

---

## Modello: RoBERTa

### 1. Average Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B0BTMY7H95` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B0BGM1PCXF` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B00V842EXS` | 0.00% | 19.50 | 50.00 | 0.05% | 0.00% |

### 2. Bandwagon Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B004DCBEHO` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B07CYZ9KS1` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B0013MTPC8` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

### 3. GenAI Attack

| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 50 | `B08S46F1YK` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 100 | `B0C9CC3NK8` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |
| 200 | `B0719D35HM` | 0.00% | 50.00 | 50.00 | 0.00% | 0.00% |

---

> **Conclusioni:** Se la posizione media nel Trust-Aware è vicina a 50 (fuori ranking) e l'Hit Rate è prossimo allo 0%, mentre nella Baseline sono alti, significa che l'Anomaly Detection ha neutralizzato con successo l'attacco, dimostrando la robustezza del sistema.
