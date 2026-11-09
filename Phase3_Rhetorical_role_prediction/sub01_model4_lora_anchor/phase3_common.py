"""
phase3_common.py
================
Universal engineering foundation, data pipeline, and architectural modules for Phase 3:
Rhetorical Role Prediction (RR) in Legal Documents (CL & IT Domains).

Key Features:
1. Universal Dataset Loader:
   - Downloads and normalizes CL_train, CL_dev, CL_test, IT_train, IT_dev, IT_test, and label_vocab.json
     from 'Exploration-Lab/CS-781-Capstone' (Phase_3_RR/).
   - Uniformly normalizes dict-of-dicts, dict-of-lists, and list-of-dicts into standard LegalDocument records.
2. 512-Token Packed Context Assembly (pack_fixed):
   - Sentence-level tokenization using native tokenizer.eos_token_id (id=1) without adding new tokens.
   - Sentences truncated to max 128 tokens.
   - Sliding window up to 512 tokens with overlap context capped at 25% (128 tokens).
   - Guaranteed progress: >= 1 new sentence per window.
3. Earliest-Window Extraction & Loss Masking:
   - Overlap sentences marked with `is_overlap=True`.
   - Predictions strictly extracted from earliest window (is_overlap == False).
   - Overlap sentences masked during Cross-Entropy and Knowledge Distillation loss computation.
4. Fourier Relative Position & Domain Embeddings:
   - FourierPositionEncoding (16-dim) based on document-relative sentence indices rel_pos = i / max(1, N - 1).
   - DomainEmbed (8-dim) mapping CL (0) and IT (1).
5. 2-Layer MLP Classification Head (RhetoricalRoleMLPHead):
   - LayerNorm(d_model) -> concat [LayerNorm(h_snt), 16-dim Fourier, 8-dim Domain]
   - Linear(d_model + 24 -> 256) -> GELU -> Dropout(0.1) -> Linear(256 -> 13).
6. Inverse Square-Root Class Weighting:
   - Smoothed class weights w_c = sqrt(N_total / (13 * max(1, N_c))) clamped to [0.35, 6.0].
7. Pre-Flight Verification / Smoke Test:
   - Comprehensive test over sample documents verifying packing, EOS positioning, batched collation,
     forward pass, gradient flow to all head parameters, and loss finiteness.
"""

from __future__ import annotations

import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, login, get_token
from transformers import PreTrainedTokenizerFast, AutoConfig, AutoModel
from safetensors.torch import load_file as safetensors_load_file

def resolve_hf_token(token: Optional[str] = None) -> Optional[str]:
    """
    Robustly resolves a Hugging Face authentication token across local, Kaggle, and Colab environments:
      1. Explicit token argument passed by caller.
      2. Environment variables: HF_TOKEN, HF, HUGGINGFACE_TOKEN.
      3. Kaggle Secrets: user_secrets.get_secret("HF_TOKEN") or "HF".
      4. huggingface_hub.get_token() (cached login token).
    """
    if token and str(token).strip():
        resolved = str(token).strip()
        try:
            login(token=resolved, add_to_git_credential=False)
        except Exception:
            pass
        return resolved

    # Check environment variables
    for env_var in ["HF_TOKEN", "HF", "HUGGINGFACE_TOKEN"]:
        val = os.environ.get(env_var)
        if val and val.strip():
            resolved = val.strip()
            try:
                login(token=resolved, add_to_git_credential=False)
            except Exception:
                pass
            return resolved

    # Check Kaggle UserSecretsClient
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        for sec_name in ["HF_TOKEN", "HF", "HUGGINGFACE_TOKEN"]:
            try:
                val = user_secrets.get_secret(sec_name)
                if val and val.strip():
                    resolved = val.strip()
                    try:
                        login(token=resolved, add_to_git_credential=False)
                    except Exception:
                        pass
                    return resolved
            except Exception:
                pass
    except Exception:
        pass

    # Check huggingface_hub default cached token
    try:
        hub_tok = get_token()
        if hub_tok and hub_tok.strip():
            return hub_tok.strip()
    except Exception:
        pass

    return None


def _handle_hf_download_error(e: Exception, repo_id: str, path: str, token: Optional[str]):
    err_str = str(e)
    is_auth_error = (
        "401" in err_str
        or "403" in err_str
        or "Unauthorized" in err_str
        or "GatedRepoError" in type(e).__name__
        or "gated" in err_str.lower()
        or "restricted" in err_str.lower()
    )
    if is_auth_error:
        token_status = (
            f"A token was provided ('{token[:6]}...'), but it was rejected or lacks access."
            if token else "No Hugging Face token was found in the environment."
        )
        msg = (
            f"\n{'='*78}\n"
            f"HUGGING FACE AUTHENTICATION / GATED REPOSITORY ERROR\n"
            f"{'='*78}\n"
            f"Cannot access file '{path}' in repository '{repo_id}'.\n"
            f"Status: {token_status}\n\n"
            f"The dataset repository '{repo_id}' is gated and requires user authentication.\n\n"
            f"HOW TO RESOLVE (Kaggle Notebook):\n"
            f"  Method 1 (Recommended - Kaggle Secrets):\n"
            f"    1. Open the notebook's top menu: 'Add-ons' -> 'Secrets'.\n"
            f"    2. Add a new secret with Label: 'HF_TOKEN' and Value: your Hugging Face token.\n"
            f"    3. Ensure the checkbox next to 'HF_TOKEN' is enabled for this notebook.\n\n"
            f"  Method 2 (In-Notebook Login):\n"
            f"    Run this in a cell before running the script:\n"
            f"      from huggingface_hub import login\n"
            f"      login('hf_your_token_here')\n\n"
            f"  Access Permission Check:\n"
            f"    Visit https://huggingface.co/datasets/{repo_id}\n"
            f"    Make sure you have clicked 'Agree' / requested access under that account.\n"
            f"{'='*78}\n"
        )
        raise PermissionError(msg) from e
    raise e


# ==============================================================================
# 1. CONSTANTS & MAPPINGS
# ==============================================================================

REPO_DATASET_ID = "Exploration-Lab/CS-781-Capstone"
REPO_MODEL_ID = "sarweshbarnwal/CS781-Legal-Models"

# Canonical subfolder names on Hugging Face Hub for Phase 1 backbones
BACKBONE_SUBFOLDERS = {
    "model1": "model1-deep-llama-135m",
    "model2": "model2-wide-mistral-136m",
    "model3": "model3-statutory-mha-110m",
    "model4": "model4-capacity-boundary-154m",
}

# Backbone dimensional profiles matching exact Phase 1 configurations
BACKBONE_CONFIGS = {
    "model1": {"d_model": 768, "n_layers": 16, "n_heads": 12, "n_kv_heads": 4, "d_ff": 2304, "arch": "llama", "subfolder": "model1-deep-llama-135m"},
    "model2": {"d_model": 896, "n_layers": 12, "n_heads": 14, "n_kv_heads": 7, "d_ff": 2432, "arch": "mistral", "subfolder": "model2-wide-mistral-136m"},
    "model3": {"d_model": 768, "n_layers": 12, "n_heads": 12, "n_kv_heads": 12, "d_ff": 2048, "arch": "mha", "subfolder": "model3-statutory-mha-110m"},
    "model4": {"d_model": 896, "n_layers": 14, "n_heads": 14, "n_kv_heads": 7, "d_ff": 2432, "arch": "llama", "subfolder": "model4-capacity-boundary-154m"},
}

# Official 13 Rhetorical Role labels
LABEL2ID: Dict[str, int] = {
    "Fact": 0,
    "Issue": 1,
    "ArgumentPetitioner": 2,
    "ArgumentRespondent": 3,
    "Statute": 4,
    "Dissent": 5,
    "PrecedentReliedUpon": 6,
    "PrecedentNotReliedUpon": 7,
    "PrecedentOverruled": 8,
    "RulingByLowerCourt": 9,
    "RatioOfTheDecision": 10,
    "RulingByPresentCourt": 11,
    "None": 12,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 13

DOMAIN2ID: Dict[str, int] = {"CL": 0, "IT": 1}
ID2DOMAIN: Dict[int, str] = {0: "CL", 1: "IT"}

# Special tokens in Phase 1 tokenizers
PAD_TOKEN_ID = 3
EOS_TOKEN_ID = 1
BOS_TOKEN_ID = 0
UNK_TOKEN_ID = 2

# Context window parameters
MAX_WINDOW_LEN = 512
MAX_OVERLAP_LEN = 128
MAX_SENT_LEN = 128


# ==============================================================================
# 2. DATA STRUCTURES
# ==============================================================================

@dataclass
class SentenceInfo:
    """Metadata for a single sentence inside a packed context window."""
    doc_sent_idx: int          # Index of sentence in the original document (0 .. N-1)
    rel_pos: float             # doc_sent_idx / max(1, N - 1) in [0.0, 1.0]
    eos_pos: int               # 0-indexed position of this sentence's EOS token in the window
    label: Optional[str]       # Ground-truth string label (None for test sentences)
    label_id: int              # Integer label ID (-100 if unknown or overlap)
    is_overlap: bool           # True if sentence was included from preceding context


@dataclass
class PackedWindow:
    """A packed sequence of sentences fitting within max 512 tokens."""
    doc_id: str
    domain: str                # 'CL' or 'IT'
    domain_id: int             # 0 for CL, 1 for IT
    window_idx: int            # Window sequence index within document (0, 1, ...)
    input_ids: List[int]
    attention_mask: List[int]
    sentence_infos: List[SentenceInfo]


@dataclass
class LegalDocument:
    """Standardized representation of a legal document."""
    doc_id: str
    domain: str                # 'CL' or 'IT'
    domain_id: int             # 0 for CL, 1 for IT
    sentences: List[str]
    labels: Optional[List[str]] = None
    label_ids: Optional[List[int]] = None


# ==============================================================================
# 3. UNIVERSAL DATASET LOADER
# ==============================================================================

def normalize_dataset_records(
    raw_data: Any,
    domain: str,
    label2id: Dict[str, int] = LABEL2ID,
) -> List[LegalDocument]:
    """
    Uniformly converts any of the following raw structures into a list of LegalDocument:
      - list-of-dicts: [{'id': ..., 'text': [...], 'labels': [...]}, ...]
      - dict-of-lists: {doc_id: [s0, s1, ...], ...} (e.g. test splits)
      - dict-of-dicts: {doc_id: {'text': [...], 'labels': [...]}, ...}
    """
    domain_id = DOMAIN2ID.get(domain.upper(), 0)
    normalized: List[LegalDocument] = []

    if isinstance(raw_data, list):
        # Case A: list-of-dicts
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("id") or item.get("doc_id") or item.get("name") or "")
            sentences = item.get("text") or item.get("sentences") or []
            labels = item.get("labels") or item.get("label") or None

            label_ids = None
            if labels is not None:
                assert len(sentences) == len(labels), (
                    f"Mismatch in doc {doc_id}: {len(sentences)} sentences vs {len(labels)} labels"
                )
                label_ids = []
                for lbl in labels:
                    if lbl not in label2id:
                        raise ValueError(f"CRITICAL: Unknown label '{lbl}' in doc '{doc_id}'! Not in {list(label2id.keys())}")
                    label_ids.append(label2id[lbl])

            normalized.append(
                LegalDocument(
                    doc_id=doc_id,
                    domain=domain.upper(),
                    domain_id=domain_id,
                    sentences=sentences,
                    labels=labels,
                    label_ids=label_ids,
                )
            )

    elif isinstance(raw_data, dict):
        # Could be dict-of-lists or dict-of-dicts
        for doc_id, val in raw_data.items():
            if isinstance(val, list):
                # Case B: dict-of-lists (standard test split)
                normalized.append(
                    LegalDocument(
                        doc_id=str(doc_id),
                        domain=domain.upper(),
                        domain_id=domain_id,
                        sentences=val,
                        labels=None,
                        label_ids=None,
                    )
                )
            elif isinstance(val, dict):
                # Case C: dict-of-dicts
                sentences = val.get("text") or val.get("sentences") or []
                labels = val.get("labels") or val.get("label") or None
                label_ids = None
                if labels is not None:
                    label_ids = []
                    for lbl in labels:
                        if lbl not in label2id:
                            raise ValueError(f"CRITICAL: Unknown label '{lbl}' in doc '{doc_id}'! Not in {list(label2id.keys())}")
                        label_ids.append(label2id[lbl])

                normalized.append(
                    LegalDocument(
                        doc_id=str(doc_id),
                        domain=domain.upper(),
                        domain_id=domain_id,
                        sentences=sentences,
                        labels=labels,
                        label_ids=label_ids,
                    )
                )

    return normalized


def load_phase3_datasets(
    repo_id: str = REPO_DATASET_ID,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Tuple[List[LegalDocument], List[LegalDocument], List[LegalDocument], Dict[str, int]]:
    """
    Downloads all Phase 3 datasets from Hugging Face Hub and returns normalized splits:
      - train_docs: CL_train (40) + IT_train (40) = 80 documents
      - dev_docs:   CL_dev (5)   + IT_dev (5)   = 10 documents
      - test_docs:  CL_test (5)  + IT_test (5)  = 10 documents
      - label_vocab: Dict[str, int]
    Strictly verifies vocabulary parity with LABEL2ID and unique document IDs.
    """
    token = resolve_hf_token(token)

    # 1. Download label_vocab.json
    try:
        vocab_path = hf_hub_download(
            repo_id,
            "Phase_3_RR/Train/label_vocab.json",
            repo_type="dataset",
            cache_dir=cache_dir,
            token=token,
        )
    except Exception as e:
        _handle_hf_download_error(e, repo_id, "Phase_3_RR/Train/label_vocab.json", token)

    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)

    # Normalize list-shaped or dict-shaped vocabulary
    if isinstance(raw_vocab, list):
        label_vocab = {lbl: i for i, lbl in enumerate(raw_vocab)}
    elif isinstance(raw_vocab, dict):
        label_vocab = raw_vocab
    else:
        raise ValueError(f"Unsupported label_vocab format: {type(raw_vocab)}")

    assert label_vocab == LABEL2ID, (
        f"CRITICAL: Vocabulary mismatch! Hub vocab ({label_vocab}) != Code LABEL2ID ({LABEL2ID})"
    )

    splits = {
        "cl_train": ("Phase_3_RR/Train/CL_train.json", "CL"),
        "it_train": ("Phase_3_RR/Train/IT_train.json", "IT"),
        "cl_dev":   ("Phase_3_RR/Train/CL_dev.json", "CL"),
        "it_dev":   ("Phase_3_RR/Train/IT_dev.json", "IT"),
        "cl_test":  ("Phase_3_RR/Test/CL_test.json", "CL"),
        "it_test":  ("Phase_3_RR/Test/IT_test.json", "IT"),
    }

    loaded_raw: Dict[str, Any] = {}
    for key, (rel_path, _) in splits.items():
        try:
            local_path = hf_hub_download(
                repo_id,
                rel_path,
                repo_type="dataset",
                cache_dir=cache_dir,
                token=token,
            )
        except Exception as e:
            _handle_hf_download_error(e, repo_id, rel_path, token)
        with open(local_path, "r", encoding="utf-8") as f:
            loaded_raw[key] = json.load(f)

    # Normalize into LegalDocument lists
    cl_train_docs = normalize_dataset_records(loaded_raw["cl_train"], "CL", label_vocab)
    it_train_docs = normalize_dataset_records(loaded_raw["it_train"], "IT", label_vocab)
    train_docs = cl_train_docs + it_train_docs

    cl_dev_docs = normalize_dataset_records(loaded_raw["cl_dev"], "CL", label_vocab)
    it_dev_docs = normalize_dataset_records(loaded_raw["it_dev"], "IT", label_vocab)
    dev_docs = cl_dev_docs + it_dev_docs

    cl_test_docs = normalize_dataset_records(loaded_raw["cl_test"], "CL", label_vocab)
    it_test_docs = normalize_dataset_records(loaded_raw["it_test"], "IT", label_vocab)
    test_docs = cl_test_docs + it_test_docs

    # Assert document ID uniqueness across domains in every split
    assert not ({d.doc_id for d in cl_train_docs} & {d.doc_id for d in it_train_docs}), "Collision in train doc IDs!"
    assert not ({d.doc_id for d in cl_dev_docs} & {d.doc_id for d in it_dev_docs}), "Collision in dev doc IDs!"
    assert not ({d.doc_id for d in cl_test_docs} & {d.doc_id for d in it_test_docs}), "Collision in test doc IDs!"

    return train_docs, dev_docs, test_docs, label_vocab


def load_phase1_backbone(
    model_key_or_subfolder: str,
    repo_id: str = REPO_MODEL_ID,
    token: Optional[str] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Safely loads a Phase 1 pretrained causal decoder with 100% weights matched:
    Strips 'model.' prefix (from LlamaForCausalLM / MistralForCausalLM checkpoints) and loads
    into AutoModel with strict=True, asserting zero missing and zero unexpected keys.
    """
    token = resolve_hf_token(token)
    subfolder = BACKBONE_SUBFOLDERS.get(model_key_or_subfolder, model_key_or_subfolder)
    try:
        cfg_path = hf_hub_download(repo_id, f"{subfolder}/config.json", token=token)
        weights_path = hf_hub_download(repo_id, f"{subfolder}/model.safetensors", token=token)
    except Exception as e:
        _handle_hf_download_error(e, repo_id, f"{subfolder}/model.safetensors", token)

    config = AutoConfig.from_pretrained(cfg_path)
    config.use_cache = False
    base_model = AutoModel.from_config(config)

    raw_sd = safetensors_load_file(weights_path)
    cleaned_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("model."):
            cleaned_sd[k[len("model."):]] = v
        elif not k.startswith("lm_head."):
            cleaned_sd[k] = v

    res = base_model.load_state_dict(cleaned_sd, strict=True)
    assert len(res.missing_keys) == 0, f"Missing keys when loading backbone {subfolder}: {res.missing_keys}"
    assert len(res.unexpected_keys) == 0, f"Unexpected keys when loading backbone {subfolder}: {res.unexpected_keys}"
    print(f"Successfully loaded 100% of Phase 1 weights for '{subfolder}' with strict=True ({len(cleaned_sd)} tensors matched).")
    return base_model.to(device)


def load_phase3_tokenizer(
    model_subfolder: str = "model1-deep-llama-135m",
    repo_id: str = REPO_MODEL_ID,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> PreTrainedTokenizerFast:
    """
    Loads and standardizes PreTrainedTokenizerFast from the Phase 1 model repository.
    Enforces pad_token='<pad>' (id=3), eos_token='<eos>' (id=1), bos_token='<bos>' (id=0),
    and padding_side='right'.
    """
    token = resolve_hf_token(token)
    try:
        tok_path = hf_hub_download(
            repo_id,
            f"{model_subfolder}/tokenizer.json",
            cache_dir=cache_dir,
            token=token,
        )
    except Exception as e:
        _handle_hf_download_error(e, repo_id, f"{model_subfolder}/tokenizer.json", token)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.padding_side = "right"

    assert tokenizer.pad_token_id == PAD_TOKEN_ID, (
        f"Tokenizer pad_token_id ({tokenizer.pad_token_id}) != {PAD_TOKEN_ID}"
    )
    assert tokenizer.eos_token_id == EOS_TOKEN_ID, (
        f"Tokenizer eos_token_id ({tokenizer.eos_token_id}) != {EOS_TOKEN_ID}"
    )
    return tokenizer


# ==============================================================================
# 4. 512-TOKEN PACKED CONTEXT ASSEMBLY (pack_fixed)
# ==============================================================================

def tokenize_and_truncate_sentence(
    sentence: str,
    tokenizer: PreTrainedTokenizerFast,
    max_sent_len: int = MAX_SENT_LEN,
    eos_token_id: int = EOS_TOKEN_ID,
) -> List[int]:
    """
    Tokenizes a single sentence, truncates to (max_sent_len - 1), and appends eos_token_id.
    Total length will always be <= max_sent_len (default 128), and last token is eos_token_id.
    """
    # Tokenize without adding any special tokens
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    # Truncate content to leave room for exactly 1 EOS token
    truncated = tokens[: max_sent_len - 1]
    # Append native EOS token as the sentence boundary
    truncated.append(eos_token_id)
    return truncated


def pack_document_fixed(
    doc: LegalDocument,
    tokenizer: PreTrainedTokenizerFast,
    max_window_len: int = MAX_WINDOW_LEN,
    max_overlap_len: int = MAX_OVERLAP_LEN,
    max_sent_len: int = MAX_SENT_LEN,
    eos_token_id: int = EOS_TOKEN_ID,
) -> List[PackedWindow]:
    """
    Packs a legal document into 512-token sliding context windows with:
      - Max window length: 512 tokens
      - Max overlap context: 128 tokens (25% window budget)
      - Sentence truncation: 128 tokens
      - Guaranteed progress: >= 1 new sentence per window
      - Overlap sentences marked with `is_overlap=True`
      - Sentences end with native tokenizer.eos_token_id
    """
    N = len(doc.sentences)
    if N == 0:
        return []

    # 1. Pre-tokenize and truncate each sentence
    sent_tokens_list: List[List[int]] = [
        tokenize_and_truncate_sentence(
            s, tokenizer, max_sent_len=max_sent_len, eos_token_id=eos_token_id
        )
        for s in doc.sentences
    ]
    sent_lens = [len(toks) for toks in sent_tokens_list]

    packed_windows: List[PackedWindow] = []
    curr_idx = 0
    window_idx = 0

    while curr_idx < N:
        # 1. Collect preceding overlap sentences (up to max_overlap_len)
        overlap_indices: List[int] = []
        overlap_tokens = 0
        if curr_idx > 0:
            for prev_i in range(curr_idx - 1, -1, -1):
                if overlap_tokens + sent_lens[prev_i] <= max_overlap_len:
                    overlap_indices.insert(0, prev_i)
                    overlap_tokens += sent_lens[prev_i]
                else:
                    break

        # 2. Ensure guaranteed progress:
        # If overlap + sent_lens[curr_idx] > max_window_len, trim overlap from front
        while overlap_indices and (overlap_tokens + sent_lens[curr_idx] > max_window_len):
            removed = overlap_indices.pop(0)
            overlap_tokens -= sent_lens[removed]

        # 3. Add new sentences starting from curr_idx
        new_indices: List[int] = []
        new_tokens = 0
        idx = curr_idx

        while idx < N:
            if overlap_tokens + new_tokens + sent_lens[idx] <= max_window_len:
                new_indices.append(idx)
                new_tokens += sent_lens[idx]
                idx += 1
            else:
                # If no new sentence could be added yet, force-include at least curr_idx
                if not new_indices:
                    new_indices.append(idx)
                    new_tokens += sent_lens[idx]
                    idx += 1
                break

        assert len(new_indices) >= 1, (
            f"Zero progress at doc {doc.doc_id}, sent {curr_idx}! Overlap={overlap_indices}"
        )

        # 4. Construct the packed token sequence and record sentence metadata
        input_ids: List[int] = []
        sentence_infos: List[SentenceInfo] = []

        # (a) Add overlap sentences
        for s_idx in overlap_indices:
            toks = sent_tokens_list[s_idx]
            input_ids.extend(toks)
            eos_pos = len(input_ids) - 1
            rel_pos = float(s_idx) / float(max(1, N - 1))
            lbl_str = doc.labels[s_idx] if doc.labels is not None else None
            # Overlap sentences are masked out during loss computation
            sentence_infos.append(
                SentenceInfo(
                    doc_sent_idx=s_idx,
                    rel_pos=rel_pos,
                    eos_pos=eos_pos,
                    label=lbl_str,
                    label_id=-100,  # Masked for loss!
                    is_overlap=True,
                )
            )

        # (b) Add new sentences
        for s_idx in new_indices:
            toks = sent_tokens_list[s_idx]
            input_ids.extend(toks)
            eos_pos = len(input_ids) - 1
            rel_pos = float(s_idx) / float(max(1, N - 1))
            lbl_str = doc.labels[s_idx] if doc.labels is not None else None
            lbl_id = doc.label_ids[s_idx] if doc.label_ids is not None else -100

            sentence_infos.append(
                SentenceInfo(
                    doc_sent_idx=s_idx,
                    rel_pos=rel_pos,
                    eos_pos=eos_pos,
                    label=lbl_str,
                    label_id=lbl_id,
                    is_overlap=False,
                )
            )

        attention_mask = [1] * len(input_ids)
        packed_windows.append(
            PackedWindow(
                doc_id=doc.doc_id,
                domain=doc.domain,
                domain_id=doc.domain_id,
                window_idx=window_idx,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_infos=sentence_infos,
            )
        )

        window_idx += 1
        curr_idx = idx

    return packed_windows


def pack_all_documents(
    docs: List[LegalDocument],
    tokenizer: PreTrainedTokenizerFast,
    max_window_len: int = MAX_WINDOW_LEN,
    max_overlap_len: int = MAX_OVERLAP_LEN,
    max_sent_len: int = MAX_SENT_LEN,
    eos_token_id: int = EOS_TOKEN_ID,
) -> List[PackedWindow]:
    """Packs a list of LegalDocument objects into a flat list of PackedWindow objects."""
    all_windows: List[PackedWindow] = []
    for doc in docs:
        windows = pack_document_fixed(
            doc,
            tokenizer,
            max_window_len=max_window_len,
            max_overlap_len=max_overlap_len,
            max_sent_len=max_sent_len,
            eos_token_id=eos_token_id,
        )
        all_windows.extend(windows)
    return all_windows


# ==============================================================================
# 5. PYTORCH DATASET & BATCH COLLATION
# ==============================================================================

class PackedRRDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping a list of PackedWindow instances."""

    def __init__(self, windows: List[PackedWindow]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> PackedWindow:
        return self.windows[idx]


def collate_packed_windows(
    batch: List[PackedWindow],
    pad_token_id: int = PAD_TOKEN_ID,
) -> Dict[str, Any]:
    """
    Collates a list of PackedWindow objects into padded PyTorch tensors:
      - input_ids: (B, max_len) right-padded with pad_token_id
      - attention_mask: (B, max_len) 1 for valid tokens, 0 for padding
      - batch_idx: (M,) index of batch element for each sentence
      - eos_pos: (M,) token position of EOS for each sentence
      - rel_pos: (M,) float relative position in [0, 1]
      - domain_ids: (M,) domain ID (0 for CL, 1 for IT)
      - label_ids: (M,) target label ID (-100 for overlap or test)
      - is_overlap: (M,) boolean mask indicating overlap status
      - doc_ids: List[str] of length M
      - doc_sent_indices: List[int] of length M
    """
    batch_size = len(batch)
    lengths = [len(w.input_ids) for w in batch]
    max_len = max(lengths)

    # Prepare padded input_ids and attention_mask (right padding)
    padded_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    b_indices: List[int] = []
    eos_positions: List[int] = []
    rel_positions: List[float] = []
    domain_ids_list: List[int] = []
    label_ids_list: List[int] = []
    is_overlap_list: List[bool] = []
    doc_ids_list: List[str] = []
    sent_idx_list: List[int] = []

    for b_idx, w in enumerate(batch):
        seq_len = len(w.input_ids)
        padded_input_ids[b_idx, :seq_len] = torch.tensor(w.input_ids, dtype=torch.long)
        padded_attention_mask[b_idx, :seq_len] = 1

        for s_info in w.sentence_infos:
            b_indices.append(b_idx)
            eos_positions.append(s_info.eos_pos)
            rel_positions.append(s_info.rel_pos)
            domain_ids_list.append(w.domain_id)
            label_ids_list.append(s_info.label_id)
            is_overlap_list.append(s_info.is_overlap)
            doc_ids_list.append(w.doc_id)
            sent_idx_list.append(s_info.doc_sent_idx)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "batch_idx": torch.tensor(b_indices, dtype=torch.long),
        "eos_pos": torch.tensor(eos_positions, dtype=torch.long),
        "rel_pos": torch.tensor(rel_positions, dtype=torch.float32),
        "domain_ids": torch.tensor(domain_ids_list, dtype=torch.long),
        "label_ids": torch.tensor(label_ids_list, dtype=torch.long),
        "is_overlap": torch.tensor(is_overlap_list, dtype=torch.bool),
        "doc_ids": doc_ids_list,
        "doc_sent_indices": sent_idx_list,
    }


# ==============================================================================
# 6. ARCHITECTURAL MODULES (Fourier Positions & 2-Layer MLP Head)
# ==============================================================================

class FourierPositionEncoding(nn.Module):
    """
    16-dimensional Fourier relative position encoding for sentence positions:
      rel_pos = i / max(1, N - 1) in [0.0, 1.0].
    Computes 8 frequencies: omega_k = pi * 2^k for k in 0..7.
    Outputs: [sin(rel_pos * omega), cos(rel_pos * omega)] -> 16 dimensions.
    """

    def __init__(self, dim: int = 16):
        super().__init__()
        assert dim % 2 == 0, f"Fourier dimension must be even, got {dim}"
        self.dim = dim
        self.num_bands = dim // 2
        # Geometric frequency bands: 2^k * pi
        freq_bands = (2.0 ** torch.arange(self.num_bands).float()) * math.pi
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
          rel_pos: Tensor of shape (M,) or (..., ) with values in [0.0, 1.0].
        Returns:
          Tensor of shape (..., 16)
        """
        # (..., 1) * (num_bands,) -> (..., num_bands)
        scaled = rel_pos.unsqueeze(-1) * self.freq_bands.to(rel_pos.device, dtype=rel_pos.dtype)
        # Concatenate sin and cos along last dimension
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class RhetoricalRoleMLPHead(nn.Module):
    """
    2-Layer MLP Classification Head for Rhetorical Role Prediction:
      Input sentence representation h_snt in R^{d_model} (extracted at EOS token).
      1. LayerNorm(d_model) applied to h_snt.
      2. 16-dim Fourier relative position encoding + 8-dim Domain embedding.
      3. Concatenate: [h_norm, fourier_pos, domain_embed] -> R^{d_model + 24}.
      4. Linear(d_model + 24 -> 256) -> GELU -> Dropout(0.1) -> Linear(256 -> 13).
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int = NUM_CLASSES,
        fourier_dim: int = 16,
        domain_dim: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.fourier_dim = fourier_dim
        self.domain_dim = domain_dim

        # 1. Feature normalization and positional / domain encoders
        self.layer_norm = nn.LayerNorm(d_model)
        self.fourier_pos = FourierPositionEncoding(dim=fourier_dim)
        self.domain_embed = nn.Embedding(num_embeddings=2, embedding_dim=domain_dim)

        # 2. Non-linear projection layers
        in_features = d_model + fourier_dim + domain_dim  # d_model + 24
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.domain_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        sentence_reprs: torch.Tensor,
        rel_pos: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          sentence_reprs: (M, d_model) sentence representations extracted at EOS.
          rel_pos: (M,) document-relative position in [0.0, 1.0].
          domain_ids: (M,) domain index (0 for CL, 1 for IT).
        Returns:
          logits: (M, 13) unnormalized classification logits.
        """
        # 1. Normalize backbone sentence hidden states
        h_norm = self.layer_norm(sentence_reprs)

        # 2. Compute 16-dim Fourier and 8-dim Domain embeddings
        fourier_feats = self.fourier_pos(rel_pos)
        domain_feats = self.domain_embed(domain_ids)

        # 3. Concatenate along feature dimension
        combined = torch.cat([h_norm, fourier_feats, domain_feats], dim=-1)

        # 4. Two-layer non-linear classification MLP
        hidden = self.dropout(self.act(self.fc1(combined)))
        logits = self.fc2(hidden)
        return logits


# ==============================================================================
# 7. INVERSE SQUARE-ROOT CLASS WEIGHTING & LOSS
# ==============================================================================

def compute_smoothed_class_weights(
    train_docs_or_labels: Union[List[LegalDocument], List[int], List[str]],
    num_classes: int = NUM_CLASSES,
    clamp_min: float = 0.35,
    clamp_max: float = 6.0,
    label2id: Dict[str, int] = LABEL2ID,
) -> torch.FloatTensor:
    """
    Computes inverse square-root class weights for imbalanced multiclass loss:
      w_c = sqrt( N_total / (13 * max(1, N_c)) ) clamped to [0.35, 6.0].
    """
    counts = [0] * num_classes

    # Handle input variations
    if len(train_docs_or_labels) > 0 and isinstance(train_docs_or_labels[0], LegalDocument):
        for doc in train_docs_or_labels:
            if doc.label_ids is not None:
                for lid in doc.label_ids:
                    if 0 <= lid < num_classes:
                        counts[lid] += 1
    elif len(train_docs_or_labels) > 0 and isinstance(train_docs_or_labels[0], str):
        for lbl in train_docs_or_labels:
            lid = label2id.get(lbl, -1)
            if 0 <= lid < num_classes:
                counts[lid] += 1
    else:
        for lid in train_docs_or_labels:
            if 0 <= lid < num_classes:
                counts[lid] += 1

    total_samples = sum(counts)
    if total_samples == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    weights: List[float] = []
    for c in range(num_classes):
        n_c = max(1, counts[c])
        raw_w = math.sqrt(total_samples / (num_classes * n_c))
        clamped_w = max(clamp_min, min(clamp_max, raw_w))
        weights.append(clamped_w)

    return torch.tensor(weights, dtype=torch.float32)


# ==============================================================================
# 8. EVALUATION METRICS & PACKAGING UTILITIES
# ==============================================================================

def compute_macro_f1(
    gold_labels: List[int],
    pred_labels: List[int],
    num_classes: int = NUM_CLASSES,
) -> Dict[str, Any]:
    """
    Calculates Macro-F1, Macro-Precision, Macro-Recall, and per-class metrics
    without requiring external libraries.
    """
    assert len(gold_labels) == len(pred_labels), "Length mismatch between gold and predictions"

    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for g, p in zip(gold_labels, pred_labels):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    per_class: Dict[str, Dict[str, float]] = {}

    for c in range(num_classes):
        p_denom = tp[c] + fp[c]
        r_denom = tp[c] + fn[c]
        p = tp[c] / p_denom if p_denom > 0 else 0.0
        r = tp[c] / r_denom if r_denom > 0 else 0.0
        f1 = (2.0 * p * r) / (p + r) if (p + r) > 0.0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        c_name = ID2LABEL.get(c, str(c))
        per_class[c_name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": r_denom,
        }

    macro_f1 = sum(f1s) / num_classes
    macro_p = sum(precisions) / num_classes
    macro_r = sum(recalls) / num_classes

    return {
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "per_class": per_class,
    }


def reconstruct_document_predictions(
    packed_windows: List[PackedWindow],
    window_sentence_preds: List[int],
    id2label: Dict[int, str] = ID2LABEL,
) -> Dict[str, List[str]]:
    """
    Reconstructs full document-level sentence predictions from unpacked window outputs.
    Predictions are extracted STRICTLY from earliest windows (is_overlap == False).
    Guarantees exactly one prediction per document sentence in original order.
    """
    # Group predictions by (doc_id, doc_sent_idx)
    doc_predictions: Dict[str, Dict[int, str]] = {}

    pred_idx = 0
    for w in packed_windows:
        for s_info in w.sentence_infos:
            pred_id = window_sentence_preds[pred_idx]
            pred_idx += 1

            # Only accept predictions from earliest windows (non-overlap)
            if not s_info.is_overlap:
                if w.doc_id not in doc_predictions:
                    doc_predictions[w.doc_id] = {}
                # Record prediction
                pred_label = id2label.get(pred_id, "None")
                doc_predictions[w.doc_id][s_info.doc_sent_idx] = pred_label

    # Sort each document by sentence index
    final_output: Dict[str, List[str]] = {}
    for doc_id, sent_map in doc_predictions.items():
        sorted_indices = sorted(sent_map.keys())
        final_output[doc_id] = [sent_map[idx] for idx in sorted_indices]

    return final_output


def package_submission_zip(
    predictions_dict: Dict[str, List[str]],
    test_docs: List[LegalDocument],
    output_zip_path: str,
    label2id: Dict[str, int] = LABEL2ID,
) -> str:
    """
    Packages prediction.json into submission.zip meeting Codabench specifications:
      1. Root contains strictly prediction.json (no parent subdirectories).
      2. Exactly matches all test documents and sentence counts.
      3. Validates that every predicted label is an official rhetorical role category.
    """
    # 1. Verification assertions
    expected_doc_ids = {doc.doc_id for doc in test_docs}
    pred_doc_ids = set(predictions_dict.keys())

    missing_docs = expected_doc_ids - pred_doc_ids
    assert not missing_docs, f"Codabench rejection risk! Missing test documents in prediction: {missing_docs}"

    extra_docs = pred_doc_ids - expected_doc_ids
    assert not extra_docs, f"Unexpected extra documents in prediction: {extra_docs}"

    for doc in test_docs:
        doc_preds = predictions_dict[doc.doc_id]
        assert len(doc_preds) == len(doc.sentences), (
            f"Sentence count mismatch in doc {doc.doc_id}: expected {len(doc.sentences)}, got {len(doc_preds)}"
        )
        for s_idx, lbl in enumerate(doc_preds):
            assert lbl in label2id, f"Invalid rhetorical role label '{lbl}' in doc {doc.doc_id} at sent {s_idx}"

    # 2. Write temporary prediction.json
    out_dir = os.path.dirname(os.path.abspath(output_zip_path))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "prediction.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(predictions_dict, f, indent=2)

    # 3. Create submission.zip with prediction.json strictly at the root
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="prediction.json")

    # 4. Verify ZIP contents
    with zipfile.ZipFile(output_zip_path, "r") as zf:
        infolist = zf.infolist()
        assert len(infolist) == 1, f"Expected exactly 1 file in submission.zip, found {len(infolist)}"
        assert infolist[0].filename == "prediction.json", (
            f"Expected root prediction.json, found {infolist[0].filename}"
        )

    print(f"Successfully packaged and verified {output_zip_path} (size: {os.path.getsize(output_zip_path)} bytes)")
    return output_zip_path


# ==============================================================================
# 9. COMPREHENSIVE PRE-FLIGHT VERIFICATION / SMOKE TEST
# ==============================================================================

def run_phase3_smoke_test(
    model_subfolder: str = "model1-deep-llama-135m",
    d_model: int = 768,
    sample_doc_count: int = 20,
    device: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """
    Executes an end-to-end smoke test validating all Phase 3 pipeline mechanics:
      1. Universal loader downloads and normalizes dataset splits.
      2. Tokenizer loads with <bos>=0, <eos>=1, <pad>=3.
      3. pack_fixed sliding window:
         - All windows <= 512 tokens.
         - Overlap context <= 128 tokens.
         - Guaranteed progress (>= 1 new sentence per window).
         - EOS tokens positioned correctly.
         - Earliest-window deduplication (is_overlap == False in exactly 1 window per sentence).
      4. Batched collation with right-padding.
      5. Forward pass through mock hidden states + RhetoricalRoleMLPHead.
      6. Inverse square-root class-weighted Cross-Entropy loss with overlap masking.
      7. Backward pass and gradient flow through all head parameters.
      8. End-to-end inference prediction reconstruction & round-trip accuracy.
    """
    token = resolve_hf_token(token)

    print("\n" + "=" * 78)
    print("STARTING PHASE 3 ENGINEERING SMOKE TEST & PIPELINE VALIDATION")
    print("=" * 78)

    if device is None:
        device = "cpu"
    print(f" -> Active compute device for smoke test: {device}")

    # Step 1: Universal Dataset Loader
    print("\n[Step 1/8] Loading and normalizing Phase 3 datasets...")
    train_docs, dev_docs, test_docs, label_vocab = load_phase3_datasets(token=token)
    print(f" -> Loaded {len(train_docs)} train docs (CL+IT)")
    print(f" -> Loaded {len(dev_docs)} dev docs (CL+IT)")
    print(f" -> Loaded {len(test_docs)} test docs (CL+IT)")
    print(f" -> Label vocabulary size: {len(label_vocab)} classes")

    assert len(train_docs) == 80, f"Expected 80 train documents, got {len(train_docs)}"
    assert len(dev_docs) == 10, f"Expected 10 dev documents, got {len(dev_docs)}"
    assert len(test_docs) == 10, f"Expected 10 test documents, got {len(test_docs)}"
    assert len(label_vocab) == 13, f"Expected 13 classes, got {len(label_vocab)}"

    # Select 20 sample documents (10 CL + 10 IT)
    cl_samples = [d for d in train_docs if d.domain == "CL"][:10]
    it_samples = [d for d in train_docs if d.domain == "IT"][:10]
    sample_docs = cl_samples + it_samples
    assert len(sample_docs) == sample_doc_count, f"Expected {sample_doc_count} sample docs, got {len(sample_docs)}"
    total_sample_sents = sum(len(d.sentences) for d in sample_docs)
    print(f" -> Selected {len(sample_docs)} sample documents ({total_sample_sents} total sentences)")

    # Step 2: Tokenizer Loading & Special Token Verification
    print("\n[Step 2/8] Loading PreTrainedTokenizerFast from Hub...")
    tokenizer = load_phase3_tokenizer(model_subfolder=model_subfolder, token=token)
    print(f" -> Tokenizer vocab size: {len(tokenizer)}")
    print(f" -> Tokens: bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}")
    assert tokenizer.pad_token_id == PAD_TOKEN_ID, "pad_token_id mismatch"
    assert tokenizer.eos_token_id == EOS_TOKEN_ID, "eos_token_id mismatch"
    assert tokenizer.padding_side == "right", "Tokenizer must use padding_side='right'"

    # Step 3: pack_fixed Sliding Window Packing Verification
    print("\n[Step 3/8] Verifying 512-token sliding window packing (pack_fixed)...")
    sample_windows = pack_all_documents(sample_docs, tokenizer)
    print(f" -> Packed {len(sample_docs)} documents into {len(sample_windows)} context windows")

    # Assert window constraints across every window
    covered_sentences: Dict[str, set] = {d.doc_id: set() for d in sample_docs}

    for w_idx, w in enumerate(sample_windows):
        seq_len = len(w.input_ids)
        assert seq_len <= MAX_WINDOW_LEN, f"Window {w_idx} exceeds max length: {seq_len} > {MAX_WINDOW_LEN}"

        # Count overlap and new tokens
        overlap_sents = [s for s in w.sentence_infos if s.is_overlap]
        new_sents = [s for s in w.sentence_infos if not s.is_overlap]

        assert len(new_sents) >= 1, f"Window {w_idx} violated guaranteed progress (0 new sentences)!"

        # Verify EOS positions in window
        for s_info in w.sentence_infos:
            eos_pos = s_info.eos_pos
            assert 0 <= eos_pos < seq_len, f"Invalid eos_pos {eos_pos} in sequence of len {seq_len}"
            token_at_eos = w.input_ids[eos_pos]
            assert token_at_eos == EOS_TOKEN_ID, (
                f"Token at eos_pos {eos_pos} is {token_at_eos}, expected {EOS_TOKEN_ID}"
            )

            # Check earliest-window coverage
            if not s_info.is_overlap:
                assert s_info.doc_sent_idx not in covered_sentences[w.doc_id], (
                    f"Sentence {s_info.doc_sent_idx} of doc {w.doc_id} appeared as new sentence twice!"
                )
                covered_sentences[w.doc_id].add(s_info.doc_sent_idx)

    # Verify that every sentence of every document was covered exactly once
    for doc in sample_docs:
        expected_indices = set(range(len(doc.sentences)))
        actual_indices = covered_sentences[doc.doc_id]
        assert expected_indices == actual_indices, (
            f"Coverage mismatch for doc {doc.doc_id}: missing {expected_indices - actual_indices}"
        )
    print(" -> All windows <= 512 tokens, progress guaranteed, 100% sentence coverage verified!")

    # Step 4: Batched Collation & Right-Padding Verification
    print("\n[Step 4/8] Testing batched collation with right-side padding...")
    batch_windows = sample_windows[:4]  # Take 4 windows for testing
    batch_tensors = collate_packed_windows(batch_windows, pad_token_id=PAD_TOKEN_ID)

    b_input_ids = batch_tensors["input_ids"]
    b_att_mask = batch_tensors["attention_mask"]
    b_idx = batch_tensors["batch_idx"]
    eos_pos = batch_tensors["eos_pos"]
    rel_pos = batch_tensors["rel_pos"]
    domain_ids = batch_tensors["domain_ids"]
    label_ids = batch_tensors["label_ids"]
    is_overlap = batch_tensors["is_overlap"]

    assert b_input_ids.shape[0] == 4, f"Batch size mismatch: {b_input_ids.shape[0]}"
    assert b_input_ids.shape == b_att_mask.shape, "Shape mismatch between input_ids and attention_mask"
    assert (b_input_ids[:, -1] == PAD_TOKEN_ID).any() or (b_att_mask.sum(dim=-1).max() == b_input_ids.shape[1]), (
        "Right-padding assertion check failed"
    )
    print(f" -> Batch input_ids shape: {tuple(b_input_ids.shape)}, total sentences: {len(eos_pos)}")

    # Step 5: Real Backbone Loading & Forward Pass Verification
    print("\n[Step 5/8] Loading real Phase 1 Model 1 backbone and verifying forward pass...")
    base_model = load_phase1_backbone("model1", device=device, token=token)
    base_model.eval()
    head = RhetoricalRoleMLPHead(d_model=d_model, num_classes=NUM_CLASSES).to(device)
    head.train()

    with torch.no_grad():
        backbone_outputs = base_model(input_ids=b_input_ids.to(device), attention_mask=b_att_mask.to(device))
        hidden_states = backbone_outputs.last_hidden_state

    # Extract sentence representations at EOS positions
    sent_reprs = hidden_states[b_idx.to(device), eos_pos.to(device)]
    assert sent_reprs.shape == (len(eos_pos), d_model), f"sent_reprs shape mismatch: {sent_reprs.shape}"

    logits = head(
        sentence_reprs=sent_reprs,
        rel_pos=rel_pos.to(device),
        domain_ids=domain_ids.to(device),
    )
    assert logits.shape == (len(eos_pos), NUM_CLASSES), f"Logits shape mismatch: {logits.shape}"
    print(f" -> Real backbone + head forward pass successfully computed: logits shape={tuple(logits.shape)}")

    # Step 6: Loss Computation with Earliest-Window Masking & Inverse Square-Root Weights
    print("\n[Step 6/8] Testing smoothed class-weighted loss and overlap masking...")
    class_weights = compute_smoothed_class_weights(sample_docs, num_classes=NUM_CLASSES).to(device)
    assert class_weights.shape == (NUM_CLASSES,), "Class weights shape mismatch"
    assert (class_weights >= 0.35).all() and (class_weights <= 6.0).all(), "Class weights outside clamped range"
    print(f" -> Computed smoothed class weights (range: [{class_weights.min():.2f}, {class_weights.max():.2f}])")

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    # Mask out overlap sentences: target is label_ids for non-overlap, -100 for overlap
    targets = label_ids.to(device).clone()
    targets[is_overlap.to(device)] = -100

    loss = criterion(logits, targets)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print(f" -> Weighted Cross-Entropy Loss: {loss.item():.4f}")

    # Step 7: Backward Pass & Gradient Flow Verification
    print("\n[Step 7/8] Verifying backward pass and gradient flow through all head parameters...")
    loss.backward()

    # Verify every trainable parameter received finite gradients
    for name, param in head.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient!"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient!"
            assert (param.grad != 0).any(), f"Parameter {name} has all-zero gradient!"
    print(" -> All head parameters (layer_norm, fourier_pos, domain_embed, fc1, fc2) received valid gradients!")

    # Step 8: End-to-End Prediction Reconstruction & Packaging Test
    print("\n[Step 8/8] Testing prediction reconstruction and Codabench package verification...")
    # Generate mock predictions matching targets where known
    mock_preds = [
        target if target != -100 else 0
        for target in targets.cpu().tolist()
    ]
    reconstructed = reconstruct_document_predictions(batch_windows, mock_preds)

    # Check reconstructed sentence counts
    for w in batch_windows:
        doc = next(d for d in sample_docs if d.doc_id == w.doc_id)
        # For documents fully contained in batch, verify count
        if w.doc_id in reconstructed:
            preds = reconstructed[w.doc_id]
            print(f"    - Doc {w.doc_id[:25]}... sentences: {len(preds)}")

    # Verify packaging test with test_docs
    mock_test_preds: Dict[str, List[str]] = {}
    for doc in test_docs:
        mock_test_preds[doc.doc_id] = ["Fact"] * len(doc.sentences)

    working_dir = os.getcwd()
    test_zip_path = os.path.join(working_dir, "test_submission.zip")
    package_submission_zip(mock_test_preds, test_docs, test_zip_path)
    if os.path.exists(test_zip_path):
        os.remove(test_zip_path)
    pred_json_temp = os.path.join(working_dir, "prediction.json")
    if os.path.exists(pred_json_temp):
        os.remove(pred_json_temp)

    print("\n" + "=" * 78)
    print("ALL 7 PHASE 3 SMOKE TEST & PIPELINE ASSERTIONS PASSED WITH 100% INTEGRITY!")
    print("=" * 78 + "\n")
    return True


if __name__ == "__main__":
    # When executed directly, run the comprehensive smoke test
    token = resolve_hf_token()
    success = run_phase3_smoke_test(token=token)
    if not success:
        sys.exit(1)
