# CS781 Capstone: Domain Adaptation & Parameter-Efficient Fine-Tuning for Indian Legal NLP

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-FFD21E?style=flat-square)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Academic Benchmark](https://img.shields.io/badge/Benchmark-IIT%20Kanpur-red?style=flat-square)](https://www.iitk.ac.in/)

> **Course:** CS781: Deep Learning for NLP / LLMs (Odd Sem 2026)  
> **Instructor:** Prof. Ashutosh Modi & Teaching Assistants (Sanjeet, Shubham, Om)  
> **Affiliation:** Department of Computer Science & Engineering, Indian Institute of Technology Kanpur (IITK)  
> **Author:** Sarwesh Barnwal (Roll Number: `250987`)  
> **Remote Model Weights:** Synchronized to Hugging Face Hub under [`sarweshbarnwal/CS781-Legal-Models`](https://huggingface.co/sarweshbarnwal/CS781-Legal-Models)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Phase 1 Foundation Models Pretrained from Scratch](#phase-1-foundation-models-pretrained-from-scratch)
3. [Nomenclature & Method Legend](#nomenclature--method-legend)
4. [Official Dataset Sources & Splits](#official-dataset-sources--splits)
5. [Phase-by-Phase Architecture & Submissions](#phase-by-phase-architecture--submissions)
   - [Phase 1: Legal Language Model Pre-training](#phase-1-legal-language-model-pre-training)
   - [Phase 2: Legal Named Entity Recognition (L-NER)](#phase-2-legal-named-entity-recognition-l-ner)
   - [Phase 3: Rhetorical Role Prediction (RRP)](#phase-3-rhetorical-role-prediction-rrp)
   - [Phase 4: Court Judgment Prediction & Explanation (CJPE)](#phase-4-court-judgment-prediction--explanation-cjpe)
   - [Phase 5: Legal Statute Identification (LSI)](#phase-5-legal-statute-identification-lsi)
   - [Phase 6: Legal Abstractive Summarization](#phase-6-legal-abstractive-summarization)
   - [Phase 7: Prior Case Retrieval (PCR)](#phase-7-prior-case-retrieval-pcr)
6. [Repository Structure](#repository-structure)
7. [Environment Setup & Reproduction Instructions](#environment-setup--reproduction-instructions)
8. [Academic References & Citation](#academic-references--citation)

---

## Project Overview

This repository contains the complete implementation, parameter-efficient adaptations (PEFT), knowledge distillation (KD) pipelines, and official submission artifacts for the **CS781 Large Language Models Capstone Benchmark** at IIT Kanpur.

The computational legal domain in India presents complex processing challenges: extreme document lengths (often exceeding 10,000 tokens), statutory cross-citations, archaic grammatical conventions, severe multi-label statutory imbalance, and strict procedural discourse conventions. This capstone project addresses these challenges through:

1. **Domain Pretraining strictly from scratch:** Four distinct causal decoder architectures (50M–300M parameters) trained on Indian Supreme Court, High Court, and Tribunal corpora. No external pretrained checkpoints (BERT, RoBERTa, LLaMA, Mistral) were used.
2. **Compulsory True Multiplicative LoRMA:** Exact implementation of Low-Rank Multiplicative Adaptation (Bihany, Patel, and Modi, 2025):
   $$y_{\text{adapted}} = y + \frac{\alpha}{r} B(A(y))$$
   with matrix $B$ initialized strictly to zero for identity preservation at step 0.
3. **Mandatory Cross-Model Knowledge Distillation:** Distilling representations from the 153.9M Model 4 teacher to the 109.5M Model 3 student via a learned dimension projection bridge $W_{\text{proj}} \in \mathbb{R}^{768 \to 896}$.
4. **Tailored Objective Functions:** Asymmetric Focal Loss (ASL) for long-tailed multi-label statute identification and hard-negative contrastive InfoNCE for precedent retrieval.
5. **Hierarchical Attention Pooling:** Handling multi-thousand-token rulings via cross-chunk attention fusion.

---

## Phase 1 Foundation Models Pretrained from Scratch

All foundation backbones were pre-trained strictly from scratch on Indian Supreme Court, High Court, and Tribunal judgments.

| Model Identifier | Architecture Style | Layers | $d_{\text{model}}$ | Attention Heads (Q / KV) | Context Length | Parameters | Core Structural Innovation |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **`model1-deep-llama-135m`** | Deep LLaMA Causal Decoder | 16 | 768 | 12 / 4 (GQA 3:1) | 1024 | 134.7M | Deep representational capacity, RMSNorm, SwiGLU FFN |
| **`model2-wide-mistral-136m`** | Wide Mistral Long-Flow | 12 | 896 | 14 / 7 (GQA 2:1) | 1536 | 136.2M | Wide channel dimension, extended 1536 context flow |
| **`model3-statutory-mha-110m`** | Statutory Precision Decoder | 12 | 768 | 12 / 12 (Full MHA) | 1024 | 109.5M | Full unshared multi-head attention for fine syntax |
| **`model4-capacity-boundary-154m`** | Peak Capacity Boundary | 14 | 896 | 14 / 14 (Full MHA) | 1024 | 153.9M | Tied embeddings, peak capacity boundary model |

---

## Nomenclature & Method Legend

| Symbol / Term | Full Name | Formal Definition / Specification |
|:---|:---|:---|
| **Model 1** | Deep LLaMA Decoder | 16-layer causal decoder with Grouped-Query Attention (3:1 GQA), SwiGLU FFN, RMSNorm (134.7M params). |
| **Model 2** | Wide Mistral Decoder | 12-layer wide-channel causal decoder with 2:1 GQA and 1536 max context (136.2M params). |
| **Model 3** | Statutory MHA Decoder | 12-layer full multi-head attention causal decoder (109.5M params). |
| **Model 4** | Capacity Boundary | 14-layer full MHA decoder with tied embeddings (153.9M params). |
| **LoRMA** | Multiplicative Adaptation | Multiplicative low-rank adaptation: $y_{\text{adapted}} = y + \frac{\alpha}{r} B(A(y))$ with $B=0$ initialization (Bihany et al., 2025). |
| **LoRA** | Low-Rank Adaptation | Additive low-rank update $W = W_0 + \frac{\alpha}{r} BA$ (Hu et al., 2022). |
| **rsLoRA** | Rank-Stabilized LoRA | LoRA scaled with $\gamma = \frac{\alpha}{\sqrt{r}}$ to prevent gradient collapse. |
| **AdaLoRA** | Adaptive LoRA | Dynamic SVD rank allocation pruning singular values during training. |
| **KD** | Knowledge Distillation | Teacher-student cross-architecture transfer with learned projection bridge $W_{\text{proj}}$. |
| **ASL** | Asymmetric Loss | Multi-label loss discounting easy negatives: $L = -y(1-p)^{\gamma_+} \log(p) - (1-y)(p_m)^{\gamma_-} \log(1-p_m)$. |

---

## Official Dataset Sources & Splits

| Phase | Task Description | Corpus Origin | Train Split | Validation / Dev Split | Test Split | Primary Metric |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| **Phase 1** | Legal LM Pretraining | Central Acts, High Courts, Supreme Court, Tribunals | $\approx$ 120M tokens | $\approx$ 6M tokens (5%) | 1,000 prompts | Perplexity / Continuity |
| **Phase 2** | Legal NER | Indian Court Judgments (L-NER Benchmark) | 35 cases | 35 cases | 35 cases | Strict Macro-F1 (13 entity classes) |
| **Phase 3** | Rhetorical Role Prediction | Commercial Law (CL) & Income Tax (IT) Judgments | 80 docs (16,845 sents) | 10 docs (2,142 sents) | 10 docs (2,197 sents) | Macro-F1 (13 rhetorical roles) |
| **Phase 4** | Court Judgment Prediction | Indian Court Judgments (Binary CJPE) | 37,387 cases | 3,505 cases | 1,517 cases | Macro-F1 (Binary Appeal Outcome) |
| **Phase 5** | Legal Statute Identification | Indian Court Judgments (Statute Identification) | 42,750 cases | 10,181 cases | 13,019 cases | Macro-F1 across 100 statute IDs |
| **Phase 6** | Abstractive Summarization | Indian Supreme Court Judgments (IN-Abs) | 6,327 cases (90%) | 703 cases (10%) | 100 cases | BERTScore F1 & ROUGE-L |
| **Phase 7** | Prior Case Retrieval | IndianKanoon Precedents (PCR) | 827 queries (4,320 cands) | 118 queries (1,023 cands) | 237 queries (1,727 cands) | Micro-F1@7 |

---

## Phase-by-Phase Architecture & Submissions

### Phase 1: Legal Language Model Pre-training
- **Task:** Domain-specific pretraining from scratch on Indian legal corpora under the 50M–300M parameter bound.
- **Architectures Evaluated:** Model 1 (Deep LLaMA GQA), Model 2 (Wide Mistral), Model 3 (Statutory MHA), Model 4 (Capacity Boundary).
- **Preliminary Explorations:** GPT-2 baseline (`exp1`), BPE tokenization ablation (`exp2`), RoPE position encoding (`exp2`), Mistral GQA scaling (`exp3`).
- **Submissions Folder:** `Phase1_legal_LM_pretraining/`

### Phase 2: Legal Named Entity Recognition (L-NER)
- **Task:** Token-level sequential BIO classification across 13 legal entity classes (`COURT`, `PETITIONER`, `RESPONDENT`, `JUDGE`, `LAWYER`, `STATUTE`, `PROVISION`, etc.) with exact character-offset boundary recovery.
- **Submissions Folder:** `Phase2_legal_NER/`
  - `S01_Model4_Full_FT/` — Full fine-tuning of Model 4.
  - `S02_Model1_LoRMA/` — Compulsory True Multiplicative LoRMA on Model 1.
  - `S03_Model1_LoRA/` — Additive LoRA baseline on Model 1.
  - `S04_Model2_Partial_FT/` — Partial layer unfreezing on Model 2.
  - `S05_Model2_Houlsby/` — Houlsby bottleneck adapter injection on Model 2.
  - `S06_Model3_Distillation/` — Mandatory Cross-Model Knowledge Distillation into Model 3 student.

### Phase 3: Rhetorical Role Prediction (RRP)
- **Task:** Sentence-level sequential discourse parsing classifying judgment sentences into rhetorical roles.
- **Submissions Folder:** `Phase3_Rhetorical_role_prediction/`
  - `sub00_model1_lorma_0context/` — Zero-context baseline with LoRMA.
  - `sub01_model4_lora_anchor/` — Model 4 LoRA anchor.
  - `sub02_model1_lorma_context/` — Bidirectional context-windowed LoRMA on Model 1.
  - `sub03_model1_rslora/` — Rank-Stabilized LoRA on Model 1.
  - `sub04_model2_adalora/` — Adaptive SVD rank allocation on Model 2.
  - `sub05_model2_ia3/` — $\text{IA}^3$ vector scaling on Model 2.
  - `sub06_model3_knowledge_distillation/` — Distilled Model 3 discourse student.
  - `sub07_master_ensemble/` — Multi-architecture probability voting ensemble (**Best Model**).

### Phase 4: Court Judgment Prediction & Explanation (CJPE)
- **Task:** Predicting binary judgment outcome ($y \in \{0, 1\}$) from long judgment texts.
- **Submissions Folder:** `Phase4_Court_judgement_prediction/`
  - `sub01_model4_full_ft/` — Full fine-tuning of Model 4.
  - `sub02_model1_true_lorma/` — Compulsory LoRMA adaptation ($r=16$) on Model 1.
  - `sub03_model2_adalora/` — AdaLoRA adaptation on Model 2.
  - `sub04_model3_houlsby/` — Houlsby bottleneck adapter on Model 3.
  - `sub06_model3_distill/` — Cross-model distilled student.
  - `sub07_model4_hierarchical_pool/` — Hierarchical cross-chunk attention pooling.
  - `sub08_master_ensemble/` — Soft Bayesian log-odds blending ensemble (**Best Model**).

### Phase 5: Legal Statute Identification (LSI)
- **Task:** Multi-label classification across 100 long-tailed statutory citations.
- **Submissions Folder:** `Phase5_Legal_statute_identification/`
  - `sub00_baseline_linear_probe/` — Frozen backbone linear probe baseline.
  - `sub01_model4_full_ft/` — Full fine-tuning with Asymmetric Focal Loss.
  - `sub01_variant_a_long/` to `sub01_variant_d_label_graph/` — Architectural ablations (chunk pooling, citation windowing, label graph priors).
  - `sub02_model1_true_lorma/` — True LoRMA multi-label adaptation.
  - `sub03_model2_adalora/` — AdaLoRA adaptation with dynamic thresholding.
  - `sub04_model1_partial_ft/` — Top-6 layer unfreezing ablation.
  - `sub05_model2_houlsby/` — Houlsby bottleneck adapter.
  - `sub06_model3_distill/` — Distilled Model 3 multi-label student.
  - `sub07_master_ensemble/` — Master ensemble with per-statute decision threshold calibration ($\theta_c \in [0.15, 0.70]$) (**Best Model**).

### Phase 6: Legal Abstractive Summarization
- **Task:** Extracting and synthesizing the judicial *ratio decidendi* into structured summaries evaluated by BERTScore F1 and ROUGE-L.
- **Submissions Folder:** `Phase6_Abstractive_summarization/`
  - `sub01_model3_full_ft_baseline/` — Full causal LM fine-tuning baseline.
  - `sub02_model1_true_lorma/` — True LoRMA adapter decoding on Model 1.
  - `sub03_model2_lora_sliding/` — Sliding-window causal LoRA on Model 2.
  - `sub04_model4_qlora_section/` — Section-conditioned QLoRA on Model 4.
  - `sub05_model1_adalora_tfidf/` — TF-IDF salience-guided AdaLoRA on Model 1.
  - `sub06_model4_to_model3_distill/` — Teacher-student distilled ratio extractor.
  - `sub07_model2_prefix_tuning/` — Prefix tuning adapter.
  - `sub08_model3_houlsby_adapter/` & `sub09_model4_pfeiffer_adapter/` — Series adapter comparisons.
  - `sub10_master_ensemble/` — Length-normalized beam search fusion (**Best Model**).

### Phase 7: Prior Case Retrieval (PCR)
- **Task:** Retrieving top-7 citing precedent case IDs given a query legal fact statement evaluated by Micro-F1@7.
- **Submissions Folder:** `Phase7_Prior_case_retrieval/`
  - `baseline_x0_lexical_bm25_tfidf/` — Lexical sparse BM25 + TF-IDF baseline.
  - `r1_model4_full_ft_teacher/` — Full fine-tuning of Model 4 bi-encoder teacher.
  - `r2_model1_lorma/` — LoRMA adapted bi-encoder on Model 1.
  - `r3_model2_partial_ft/` — Partial fine-tuning on Model 2.
  - `sub01_model4_full_ft_hierarchical/` — Two-Tower InfoNCE Bi-Encoder with Hierarchical Cross-Chunk Attention.
  - `sub02_model1_true_lorma_hardneg/` — True LoRMA with BM25 hard-negative mining ($\tau=0.05$).
  - `sub03_model2_qlora_attention_pool/` — QLoRA with attention pooling on Model 2.
  - `sub04_model1_adalora_margin_mse/` — Margin-MSE objective with AdaLoRA.
  - `sub05_model2_houlsby_cosent/` — CoSENT ranking loss with Houlsby adapter.
  - `sub06_model4_to_model3_distill/` — Representation distillation into compact Model 3 bi-encoder.
  - `sub07_model4_poly_encoder_multiaspect/` — Poly-encoder with multi-aspect context codes.
  - `sub08_master_ensemble/` — Dense-Sparse BM25 + Multi-Bi-Encoder reciprocal rank fusion (**Best Model**).

---

## Repository Structure

```text
CS781-Legal_Language_Modelling/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── Phase1_legal_LM_pretraining/
│   ├── M1_deep_llama_135m/                  # Notebook + submission.zip + configs
│   ├── M2_wide_mistral_136m/                # Notebook + submission.zip + configs
│   ├── M3_statutory_mha_110m/               # Notebook + submission.zip + configs
│   ├── M4_capacity_boundary_154m/           # Notebook + submission.zip + configs
│   ├── exp1_legal_gpt2_32k/                 # Preliminary experiment
│   ├── exp2_legal_bpe_32k/                  # Preliminary experiment
│   ├── exp2_legal_llama_rope_32k/           # Preliminary experiment
│   └── exp3_legal_mistral_gqa_32k/          # Preliminary experiment
│
├── Phase2_legal_NER/
│   ├── S01_Model4_Full_FT/                  # Notebook + submission.zip + val_metrics.json
│   ├── S02_Model1_LoRMA/                    # Compulsory True LoRMA
│   ├── S03_Model1_LoRA/                     # Additive LoRA baseline
│   ├── S04_Model2_Partial_FT/               # Partial fine-tuning
│   ├── S05_Model2_Houlsby/                  # Bottleneck adapter
│   └── S06_Model3_Distillation/             # Mandatory KD Student
│
├── Phase3_Rhetorical_role_prediction/
│   ├── sub00_model1_lorma_0context/
│   ├── sub01_model4_lora_anchor/
│   ├── sub02_model1_lorma_context/
│   ├── sub03_model1_rslora/
│   ├── sub04_model2_adalora/
│   ├── sub05_model2_ia3/
│   ├── sub06_model3_knowledge_distillation/
│   └── sub07_master_ensemble/
│
├── Phase4_Court_judgement_prediction/
│   ├── sub01_model4_full_ft/
│   ├── sub02_model1_true_lorma/
│   ├── sub03_model2_adalora/
│   ├── sub04_model3_houlsby/
│   ├── sub06_model3_distill/
│   ├── sub07_model4_hierarchical_pool/
│   └── sub08_master_ensemble/
│
├── Phase5_Legal_statute_identification/
│   ├── sub00_baseline_linear_probe/
│   ├── sub01_model4_full_ft/
│   ├── sub01_variant_a_long/ ... sub01_variant_d_label_graph/
│   ├── sub02_model1_true_lorma/
│   ├── sub03_model2_adalora/
│   ├── sub04_model1_partial_ft/
│   ├── sub05_model2_houlsby/
│   ├── sub06_model3_distill/
│   └── sub07_master_ensemble/
│
├── Phase6_Abstractive_summarization/
│   ├── sub01_model3_full_ft_baseline/
│   ├── sub02_model1_true_lorma/
│   ├── sub03_model2_lora_sliding/
│   ├── sub04_model4_qlora_section/
│   ├── sub05_model1_adalora_tfidf/
│   ├── sub06_model4_to_model3_distill/
│   ├── sub07_model2_prefix_tuning/
│   ├── sub08_model3_houlsby_adapter/
│   ├── sub09_model4_pfeiffer_adapter/
│   └── sub10_master_ensemble/
│
└── Phase7_Prior_case_retrieval/
    ├── baseline_x0_lexical_bm25_tfidf/
    ├── r1_model4_full_ft_teacher/
    ├── r2_model1_lorma/
    ├── r3_model2_partial_ft/
    ├── sub01_model4_full_ft_hierarchical/
    ├── sub02_model1_true_lorma_hardneg/
    ├── sub03_model2_qlora_attention_pool/
    ├── sub04_model1_adalora_margin_mse/
    ├── sub05_model2_houlsby_cosent/
    ├── sub06_model4_to_model3_distill/
    ├── sub07_model4_poly_encoder_multiaspect/
    └── sub08_master_ensemble/
```

---

## Environment Setup & Reproduction Instructions

### 1. Environment Setup
```bash
git clone https://github.com/sarweshbarnwal/CS781-Legal_Language_Modelling.git
cd CS781-Legal_Language_Modelling
pip install -r requirements.txt
```

### 2. Loading Weights from Hugging Face Hub
All pre-trained backbones and fine-tuned checkpoints are publicly accessible via `huggingface_hub`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Phase 1 Foundation Model (Model 1)
model = AutoModelForCausalLM.from_pretrained(
    "sarweshbarnwal/CS781-Legal-Models",
    subfolder="model1-deep-llama-135m"
)
tokenizer = AutoTokenizer.from_pretrained(
    "sarweshbarnwal/CS781-Legal-Models",
    subfolder="model1-deep-llama-135m"
)
```

### 3. Running Reproduction Notebooks
Every submission folder (`PhaseX_.../subXX_.../`) contains a self-contained Jupyter notebook (`.ipynb`) configured for Dual-T4 Kaggle GPU or Google Colab environments. Open any notebook and execute from top to bottom.

---

## Academic References & Citation

1. **IL-TUR Benchmark**: *IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning* (arXiv:2407.05399).
2. **LoRMA**: Bihany, B., Patel, A., and Modi, A. (2025). *LORMA: Low-Rank Multiplicative Adaptation of Large Language Models*.
3. **LoRA**: Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
4. **Knowledge Distillation**: Hinton, G., Vinyals, O., and Dean, J. (2015). *Distilling the Knowledge in a Neural Network*, NeurIPS Deep Learning Workshop.
5. **Asymmetric Loss (ASL)**: Ridnik, T., et al. (2021). *Asymmetric Loss For Multi-Label Classification*, ICCV 2021.
6. **AdaLoRA**: Zhang, Q., et al. (2023). *Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, ICLR 2023.

---

## Authors & Acknowledgments

- **Author:** Sarwesh Barnwal (Roll: `250987`), Department of Computer Science & Engineering, IIT Kanpur.
- **Supervisor:** Prof. Ashutosh Modi, IIT Kanpur.
- **Teaching Assistants:** Sanjeet, Shubham, Om.
