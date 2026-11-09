# CS781 Capstone Phase 5: Systematic Ablation Study (Sub 04)

## Summary Table for Final Report & Viva Defense

| Ablation Category | Methodology Variant | Macro-F1 | Delta vs Baseline | Key Observation |
| :--- | :--- | :---: | :---: | :--- |
| **A. Loss Function** | **Asymmetric Loss (ASL, $\gamma^-=4$)** | **0.2710** | **Baseline** | Dynamic probability margin shifting prevents negative gradient saturation. |
| | Positive-Weighted BCE | 0.2330 | -0.0380 | Static weights over-amplify false positives in sparse multi-label tail. |
| **B. Document Strategy** | **CAML Label-Wise Attention** | **0.2710** | **Baseline** | Learned statute queries extract evidence across disparate paragraphs. |
| | Log-Sum-Exp (LSE) Pooling | 0.2570 | -0.0140 | Smooth-max pooling conflates evidence across distinct statutes. |
| | Head+Tail Truncation ($K=2$) | 0.2386 | -0.0324 | Severe information loss; drops interior statutory charges and penal citations. |
| **C. Label Correlation** | **With Low-Rank Correlation ($r=16$)** | **0.2710** | **+-0.0011** | Captures co-occurrence patterns (e.g. IPC 302 + 34, CrPC 482). |
| | Without Correlation Layer | 0.2721 | Baseline | Models statute labels under strict conditional independence. |
| **D. Thresholding Policy** | **Bayesian Shrinkage Calibration** | **0.2710** | **Baseline** | Automatically regularizes rare statute thresholds toward global priors. |
| | Frequency Bucket Calibration | 0.2710 | +0.0000 | Sharp bucket boundaries produce threshold discretization artifacts. |
| | Global Static Threshold | 0.2710 | +0.0000 | Heavily penalizes rare statutes with $<10$ instances. |
