# Adversarial Evaluation Report

**Data:** 2026-06-05 15:13

Questo report confronta la resilienza del sistema Baseline rispetto al Trust-Aware sotto attacchi di Data Poisoning.

## Modello: SBERT

### 1. Average Attack
- **Target Item:** `B0085OU6Z2`
- **Bot Detection Rate:** 0.00% (bot bloccati dall'Isolation Forest)

| Metrica | Baseline (Poisoned) | Trust-Aware (Poisoned) |
|---------|-------------------|------------------------|
| Posizione Media Target | 18.50 | 50.00 |
| Hit Rate@20 Target | 0.05% | 0.00% |

### 2. Bandwagon Attack
- **Target Item:** `B00FMRV7OY`
- **Bot Detection Rate:** 0.00% (bot bloccati dall'Isolation Forest)

| Metrica | Baseline (Poisoned) | Trust-Aware (Poisoned) |
|---------|-------------------|------------------------|
| Posizione Media Target | 50.00 | 50.00 |
| Hit Rate@20 Target | 0.00% | 0.00% |

---
## Modello: RoBERTa

### 1. Average Attack
- **Target Item:** `B009A6P2VC`
- **Bot Detection Rate:** 0.00% (bot bloccati dall'Isolation Forest)

| Metrica | Baseline (Poisoned) | Trust-Aware (Poisoned) |
|---------|-------------------|------------------------|
| Posizione Media Target | 17.00 | 50.00 |
| Hit Rate@20 Target | 0.20% | 0.00% |

### 2. Bandwagon Attack
- **Target Item:** `B01NGZ69P6`
- **Bot Detection Rate:** 0.00% (bot bloccati dall'Isolation Forest)

| Metrica | Baseline (Poisoned) | Trust-Aware (Poisoned) |
|---------|-------------------|------------------------|
| Posizione Media Target | 50.00 | 50.00 |
| Hit Rate@20 Target | 0.00% | 0.00% |

---
> **Conclusioni:** Se la posizione media nel Trust-Aware è vicina a 50 (fuori ranking) e l'Hit Rate è prossimo allo 0%, mentre nella Baseline sono alti, significa che l'Anomaly Detection ha neutralizzato con successo l'attacco, dimostrando la robustezza del sistema.
