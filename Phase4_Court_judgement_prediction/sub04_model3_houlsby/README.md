---
base_model: sarweshbarnwal/CS781-Legal-Models
tags:
- legal
- legal-nlp
- court-judgment-prediction
- ildc
- adapters
- houlsby
---

# Phase 4 Submission 04: Model 3 Statutory Precision Full-MHA Decoder (Houlsby Bottleneck Adapters)
- **Base Model:** `sarweshbarnwal/CS781-Legal-Models` (subfolder: `model3-statutory-mha-110m`)
- **Task:** Court Judgment Prediction (CJPE - Binary Classification)
- **Role:** Adapter Family Evaluation (residual bottleneck modules on Full MHA)
- **Adaptation Method:** Houlsby Residual Bottleneck Adapters ($d_{\text{bottleneck}}=56$, SiLU non-linearity)
- **Adapter Locations:** Injected post-Self-Attention (`o_proj`) and post-Feed-Forward (`down_proj`) across all 12 layers
- **Best Dev Macro-F1:** 0.6317 (Optimal Decision Threshold tau*: 0.39)
- **Trainable Parameters:** 2,677,825 / 111,614,016 (2.40%)
- **Test Predictions:** Available in `prediction.json` and packaged in `submission.zip`
