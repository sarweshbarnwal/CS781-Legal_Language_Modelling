---
base_model: sarweshbarnwal/CS781-Legal-Models
tags:
- legal
- legal-nlp
- court-judgment-prediction
- ildc
- peft
- adalora
---

# Phase 4 Submission 03: Model 2 Wide Mistral Style Decoder (AdaLoRA Dynamic Rank Allocation)
- **Base Model:** `sarweshbarnwal/CS781-Legal-Models` (subfolder: `model2-wide-mistral-136m`)
- **Task:** Court Judgment Prediction (CJPE - Binary Classification)
- **Role:** Adaptive Budget Allocation (SVD triplet singular value dynamic pruning)
- **Adaptation Method:** AdaLoRA (initial rank $r=24 	o$ target rank $r=12$ on `q_proj, k_proj, v_proj, o_proj`)
- **Regularization:** Frobenius norm SVD orthogonality regularization ($\mathcal{L}_{\text{orth}}$, weight: 0.5)
- **Best Dev Macro-F1:** 0.6024 (Optimal Decision Threshold tau*: 0.56)
- **Trainable Parameters:** 2,499,969 / 137,849,904 (1.814%)
- **Test Predictions:** Available in `prediction.json` and packaged in `submission.zip`
