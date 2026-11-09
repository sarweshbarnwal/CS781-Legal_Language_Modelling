---
base_model: sarweshbarnwal/CS781-Legal-Models
tags:
- legal
- legal-nlp
- court-judgment-prediction
- ildc
---

# Phase 4 Submission 01: Model 4 Full Fine-Tuning Anchor & KD Teacher
- **Base Model:** `sarweshbarnwal/CS781-Legal-Models` (subfolder: `model4-capacity-boundary-154m`)
- **Task:** Court Judgment Prediction (CJPE - Binary Classification)
- **Role:** Performance Upper-Bound Anchor & Knowledge Distillation Teacher (for Sub 06)
- **Adaptation Method:** Full Parameter Fine-Tuning (100% backbone unfrozen)
- **Best Dev Macro-F1:** 0.3333 (Optimal Decision Threshold tau*: 0.10)
- **Trainable Parameters:** 154,629,761 / 153,937,280 (100.0%)
- **Teacher Artifacts:** Exported soft targets `sub01_teacher_logits.pt` at temperature T=2.0
