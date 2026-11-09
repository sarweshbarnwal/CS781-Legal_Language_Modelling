---
base_model: sarweshbarnwal/CS781-Legal-Models
tags:
- legal
- legal-nlp
- court-judgment-prediction
- ildc
---

# Phase 4 Submission 02: Model 1 True Multiplicative LoRMA (Compulsory PEFT)
- **Base Model:** `sarweshbarnwal/CS781-Legal-Models` (subfolder: `model1-deep-llama-135m`)
- **Task:** Court Judgment Prediction (CJPE - Binary Classification)
- **Role:** Compulsory LoRMA PEFT Submission
- **Adaptation Method:** True Multiplicative LoRMA (r=16, alpha=32 on attention output dimension)
- **Multi-Chunk Pipeline:** Last K=4 non-overlapping 1024-token chunks with Chunk Attention Pooling
- **Best Dev Macro-F1:** 0.6477 (Optimal Decision Threshold tau*: 0.47)
- **Trainable Parameters:** 1,642,241 / 135,750,400 (1.210%)
