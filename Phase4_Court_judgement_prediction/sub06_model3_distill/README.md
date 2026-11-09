---
base_model: sarweshbarnwal/CS781-Legal-Models
tags:
- legal
- legal-nlp
- court-judgment-prediction
- ildc
- knowledge-distillation
- student-model
---

# Phase 4 Submission 06: Model 3 Statutory Precision Full-MHA (Knowledge Distillation Student)
- **Base Student Model:** `sarweshbarnwal/CS781-Legal-Models` (subfolder: `model3-statutory-mha-110m`)
- **Teacher Model:** Model 4 Peak Capacity Boundary Decoder (153.9M, Sub 01 Anchor)
- **Task:** Court Judgment Prediction (CJPE - Binary Classification)
- **Role:** Compulsory Knowledge Distillation (validated capacity reduction: 153.9M -> 109.5M, 28.8% reduction)
- **Adaptation Method:** Full Parameter Student Training with 0.4 * BCE + 0.6 * T^2 * MSE distillation loss (T=2.0)
- **Best Dev Macro-F1:** 0.6394 (Optimal Decision Threshold tau*: 0.46)
- **Trainable Parameters:** 110,123,521 (100.0% of student)
- **Test Predictions:** Available in `prediction.json` and packaged in `submission.zip`
