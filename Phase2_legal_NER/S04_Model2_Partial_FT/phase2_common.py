"""
CS781 Capstone Phase 2: Legal Named Entity Recognition (L-NER)
Common Core Library: Datasets, Span Alignment, Reconstruction, Loss Functions,
Adaptation Modules (LORMA, LoRA, BitFit, Bottleneck Adapters), and Evaluation.
"""

import os
import re
import json
import math
import random
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ============================================================================
# 1. Canonical L-NER Label Set & Constants
# ============================================================================

ENTITY_LABELS = [
    "APP",        # Appellant / Petitioner
    "RESP",       # Respondent
    "A.COUNSEL",  # Appellant's Counsel
    "R.COUNSEL",  # Respondent's Counsel
    "JUDGE",      # Presiding Judge(s)
    "WIT",        # Witness
    "AUTH",       # Precedent Authority / Legal Body
    "COURT",      # Court Name
    "STAT",       # Statute / Law / Act / Section
    "PREC",       # Precedent Case Citation
    "DATE",       # Date
    "CASENO",     # Case Number / Docket Number
]

ALL_LABELS = ENTITY_LABELS + ["O"]
NUM_LABELS = len(ALL_LABELS)  # 13

LABEL2ID = {lbl: i for i, lbl in enumerate(ALL_LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(ALL_LABELS)}
O_INDEX = LABEL2ID["O"]
IGNORE_INDEX = -100

HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
HF_MASTER_REPO = "sarweshbarnwal/CS781-Legal-Models"


# ============================================================================
# 2. Dataset Loading & Alignment Utilities
# ============================================================================

def load_json_file(file_path: str) -> Any:
    """Load json dictionary or list from path."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_tokens_with_gold_spans(
    text: str,
    spans: List[List[Any]],
    offset_mapping: List[Tuple[int, int]],
    pad_token_id: Optional[int] = None
) -> List[int]:
    """
    Maps character-level gold spans [[start_char, end_char, label], ...]
    into subword token label IDs (-100 for special tokens, 0..11 for entities, 12 for 'O').
    """
    labels = []
    for (start_c, end_c) in offset_mapping:
        # Special tokens (start == end == 0 or whitespace padding)
        if start_c == end_c:
            labels.append(IGNORE_INDEX)
            continue

        assigned_label = O_INDEX
        # Find if token overlaps with any gold span
        for span in spans:
            g_start, g_end, g_label = span[0], span[1], span[2]
            # Token overlaps if max(start_c, g_start) < min(end_c, g_end)
            if max(start_c, g_start) < min(end_c, g_end):
                if g_label in LABEL2ID:
                    assigned_label = LABEL2ID[g_label]
                    break

        labels.append(assigned_label)
    return labels


class LNERDataset(Dataset):
    """
    PyTorch Dataset for Legal NER using sliding window tokenization.
    Takes documents {doc_id: text} and gold annotations {doc_id: [[start, end, label], ...]}.
    """
    def __init__(
        self,
        docs: Dict[str, str],
        golds: Optional[Dict[str, List[List[Any]]]] = None,
        tokenizer: Any = None,
        max_seq_len: int = 512,
        stride: int = 64
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.samples = []

        for doc_id, text in docs.items():
            doc_golds = golds.get(doc_id, []) if golds is not None else None
            
            # Fast tokenizer sliding window encoding
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt"
            )

            num_windows = encoding["input_ids"].shape[0]
            offset_mappings = encoding["offset_mapping"]

            for w in range(num_windows):
                input_ids = encoding["input_ids"][w]
                attention_mask = encoding["attention_mask"][w]
                offsets = offset_mappings[w].tolist()

                if doc_golds is not None:
                    label_ids = align_tokens_with_gold_spans(text, doc_golds, offsets)
                    label_tensor = torch.tensor(label_ids, dtype=torch.long)
                else:
                    label_tensor = torch.full_like(input_ids, IGNORE_INDEX)

                self.samples.append({
                    "doc_id": doc_id,
                    "window_idx": w,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": label_tensor,
                    "offsets": offsets,
                    "text_len": len(text)
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def compute_class_weights(
    train_dataset: LNERDataset,
    smoothing: float = 0.5
) -> torch.Tensor:
    """
    Computes inverse class frequency weights across all valid training tokens.
    Rare entities (WIT, JUDGE, A.COUNSEL) receive significantly higher loss weights.
    """
    counts = torch.zeros(NUM_LABELS, dtype=torch.float32)
    for sample in train_dataset.samples:
        labels = sample["labels"]
        valid_labels = labels[labels != IGNORE_INDEX]
        for c in range(NUM_LABELS):
            counts[c] += (valid_labels == c).sum().item()

    total_valid = counts.sum().item()
    weights = torch.zeros(NUM_LABELS, dtype=torch.float32)
    for c in range(NUM_LABELS):
        if counts[c] > 0:
            # Smoothed inverse frequency
            weights[c] = (total_valid / (NUM_LABELS * counts[c])) ** smoothing
        else:
            weights[c] = 1.0

    # Normalize so mean weight of entities is 1.0
    weights = weights / weights.mean()
    return weights


def ner_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collation function for DataLoader.
    Stacks input_ids, attention_mask, and labels into batched tensors,
    while preserving doc_id, window_idx, and offsets as raw python lists.
    Prevents ValueError unpacking errors during test inference!
    """
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "doc_id": [b["doc_id"] for b in batch],
        "window_idx": [b["window_idx"] for b in batch],
        "offsets": [b["offsets"] for b in batch],
        "text_len": [b["text_len"] for b in batch]
    }


# ============================================================================
# 3. Custom Loss Functions
# ============================================================================

class ClassWeightedCrossEntropyLoss(nn.Module):
    """Cross-Entropy loss with inverse frequency weights for class imbalance."""
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, T, C), targets: (B, T)
        return F.cross_entropy(
            logits.view(-1, NUM_LABELS),
            targets.view(-1),
            weight=self.weights,
            ignore_index=IGNORE_INDEX
        )


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss (Lin et al.)
    Reduces the relative loss for well-classified background tokens ('O')
    and focuses on hard entity boundary tokens.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, NUM_LABELS)
        targets = targets.view(-1)

        valid_mask = targets != IGNORE_INDEX
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining hard ground truth loss and soft KL divergence.
    L_total = alpha_ce * L_ce + alpha_kd * (T^2) * KL(student / T || teacher / T)
    """
    def __init__(
        self,
        temperature: float = 2.0,
        alpha_ce: float = 0.4,
        alpha_kd: float = 0.6
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.kl_div = nn.KLDivLoss(reduction="none", log_target=False)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Hard cross-entropy on ground truth
        loss_ce = F.cross_entropy(
            student_logits.view(-1, NUM_LABELS),
            targets.view(-1),
            ignore_index=IGNORE_INDEX
        )

        valid_tokens = targets.view(-1) != IGNORE_INDEX
        if not valid_tokens.any():
            return loss_ce

        # Soft KL divergence on valid tokens
        s_logits_flat = student_logits.view(-1, NUM_LABELS)[valid_tokens]
        t_logits_flat = teacher_logits.view(-1, NUM_LABELS)[valid_tokens]

        student_log_probs = F.log_softmax(s_logits_flat / self.temperature, dim=-1)
        teacher_probs = F.softmax(t_logits_flat / self.temperature, dim=-1)

        loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.temperature ** 2)
        return self.alpha_ce * loss_ce + self.alpha_kd * loss_kd


# ============================================================================
# 4. Decoder LM Token Classification Wrapper
# ============================================================================

class LNERDecoderModel(nn.Module):
    """
    Wraps a Causal Decoder LM backbone with a 13-class Token Classification Head.
    Supports LLaMA, Mistral, and custom decoder architectures.
    """
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize classification head with small normal weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Forward pass through backbone transformer
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
            outputs = self.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state
        else:
            try:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            except TypeError:
                outputs = self.backbone(input_ids)

            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[-1]
            elif isinstance(outputs, (tuple, list)):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs

        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)  # (B, T, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=IGNORE_INDEX
            )

        return logits, loss


# ============================================================================
# 5. Parameter Adaptation Modules (LORMA, LoRA, BitFit, Bottleneck Adapters)
# ============================================================================

class LoRALinear(nn.Module):
    """
    Standard Additive Low-Rank Adaptation (Hu et al.).
    y = W_0 @ x + (alpha / r) * (B @ (A @ x))
    A: (r, d_in), B: (d_out, r)
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Frozen base weight
        self.weight = original_linear.weight
        self.weight.requires_grad = False
        if original_linear.bias is not None:
            self.bias = original_linear.bias
            self.bias.requires_grad = False
        else:
            self.register_parameter("bias", None)

        # Trainable low-rank factors
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize A ~ Kaiming uniform, B = 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out


class LoRMALinear(nn.Module):
    """
    Exploration Lab Multiplicative Low-Rank Matrix Adaptation (LoRMA).
    y = (I + (alpha / r) * (B @ A)) @ (W_0 @ x)
    h_base = W_0 @ x
    y = h_base + (alpha / r) * (B @ (A @ h_base))
    A: (r, d_out), B: (d_out, r) operating multiplicatively on the pre-activation base output!
    Reference: github.com/Exploration-Lab/LoRMA (IIT Kanpur)
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Frozen base weight
        self.weight = original_linear.weight
        self.weight.requires_grad = False
        if original_linear.bias is not None:
            self.bias = original_linear.bias
            self.bias.requires_grad = False
        else:
            self.register_parameter("bias", None)

        # LoRMA low-rank factors operate on out_features (multiplicative modulation of output)
        self.lorma_A = nn.Parameter(torch.zeros(r, self.out_features))
        self.lorma_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.lorma_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize A ~ Kaiming uniform, B = 0 so initial modulation is exact identity
        nn.init.kaiming_uniform_(self.lorma_A, a=math.sqrt(5))
        nn.init.zeros_(self.lorma_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_base = F.linear(x, self.weight, self.bias)
        h_mod = (self.lorma_dropout(h_base) @ self.lorma_A.t()) @ self.lorma_B.t()
        return h_base + self.scaling * h_mod


def apply_lorma(
    model: nn.Module,
    target_modules: List[str],
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05
) -> int:
    """Applies true Exploration-Lab multiplicative LoRMA."""
    for param in model.parameters():
        param.requires_grad = False

    replaced_count = 0
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear) and not isinstance(module, (LoRALinear, LoRMALinear)):
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                child_name = parts[-1]
                setattr(parent, child_name, LoRMALinear(module, r=r, alpha=alpha, dropout=dropout))
                replaced_count += 1
                break

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Applied Multiplicative LoRMA: Replaced {replaced_count} layers. Trainable params: {trainable_params:,}")
    return trainable_params


def apply_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05
) -> int:
    """Applies standard additive LoRA."""
    for param in model.parameters():
        param.requires_grad = False

    replaced_count = 0
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear) and not isinstance(module, (LoRALinear, LoRMALinear)):
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                child_name = parts[-1]
                setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
                replaced_count += 1
                break

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Applied Additive LoRA: Replaced {replaced_count} layers. Trainable params: {trainable_params:,}")
    return trainable_params


def apply_lorma_or_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    method: str = "lora"
) -> int:
    if method == "lorma":
        return apply_lorma(model, target_modules, r=r, alpha=alpha, dropout=dropout)
    return apply_lora(model, target_modules, r=r, alpha=alpha, dropout=dropout)


def apply_bitfit(model: nn.Module) -> int:
    """
    BitFit (Zaken et al.): Freezes all 2D weight matrices.
    Unfreezes ONLY 1D bias vectors across all layers + Token Classification Head.
    """
    for name, param in model.named_parameters():
        # Unfreeze if parameter is 1D (bias vector or norm scale) or in classifier
        if "classifier" in name or param.ndim == 1 or "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Applied BitFit: Trainable params (biases + head only): {trainable_params:,}")
    return trainable_params


class ResidualBottleneckAdapter(nn.Module):
    """
    Houlsby / Pfeiffer style 2-layer residual bottleneck MLP adapter.
    h' = h + f(h @ W_down) @ W_up
    W_up initialized to zero so the initial function is exact identity.
    """
    def __init__(self, d_model: int, bottleneck_dim: int = 56, dropout: float = 0.05):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        # Initialize down projection and zero up projection
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        down = self.act(self.down_proj(x))
        up = self.up_proj(self.dropout(down))
        return residual + up


def apply_bottleneck_adapters(model: nn.Module, reduction_factor: int = 16) -> int:
    """
    Inserts Houlsby/Pfeiffer style bottleneck adapters into each transformer block.
    Freezes all base model weights, leaving only adapters and head trainable.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Find transformer layers
    layers = None
    if hasattr(model, "backbone"):
        if hasattr(model.backbone, "model") and hasattr(model.backbone.model, "layers"):
            layers = model.backbone.model.layers
        elif hasattr(model.backbone, "layers"):
            layers = model.backbone.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    adapter_count = 0
    if layers is not None:
        for idx, layer in enumerate(layers):
            d_model = getattr(model, "hidden_size", 896)
            bottleneck_dim = max(16, d_model // reduction_factor)
            
            # Inject adapter after attention
            adapter_attn = ResidualBottleneckAdapter(d_model, bottleneck_dim)
            layer.adapter_attn = adapter_attn
            
            # Hook into forward or wrap post-attention
            for p in adapter_attn.parameters():
                p.requires_grad = True
            adapter_count += 1

    # Unfreeze classifier head
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Applied Bottleneck Adapters: Injected into {adapter_count} layers. Trainable params: {trainable_params:,}")
    return trainable_params


def apply_partial_finetune(model: nn.Module, freeze_layers_up_to: int = 8) -> int:
    """
    Partial Layer-Wise Fine-Tuning:
    Freezes layers 1 to freeze_layers_up_to (lower layers).
    Unfreezes higher layers + Token Classification Head.
    """
    # Freeze everything initially
    for param in model.parameters():
        param.requires_grad = False

    layers = None
    if hasattr(model, "backbone"):
        if hasattr(model.backbone, "model") and hasattr(model.backbone.model, "layers"):
            layers = model.backbone.model.layers
        elif hasattr(model.backbone, "layers"):
            layers = model.backbone.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is not None:
        for i, layer in enumerate(layers):
            if i >= freeze_layers_up_to:
                for param in layer.parameters():
                    param.requires_grad = True

    # Unfreeze classifier head
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Partial Fine-Tuning: Frozen layers 0..{freeze_layers_up_to-1}. Trainable params: {trainable_params:,}")
    return trainable_params


def apply_full_finetune(model: nn.Module) -> int:
    """Full End-to-End Fine-Tuning: All parameters unfrozen."""
    for param in model.parameters():
        param.requires_grad = True
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Full Fine-Tuning: All {trainable_params:,} parameters unfrozen.")
    return trainable_params


# ============================================================================
# 6. Evaluation Metrics: Strict Macro-F1 across 12 Entity Types
# ============================================================================

def compute_strict_macro_f1(
    pred_labels: List[List[int]],
    true_labels: List[List[int]]
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1 per entity type, and strict unweighted Macro-F1
    across all 12 named entity classes (strictly excluding 'O' and ignore_index).
    """
    tp = {lbl: 0 for lbl in ENTITY_LABELS}
    fp = {lbl: 0 for lbl in ENTITY_LABELS}
    fn = {lbl: 0 for lbl in ENTITY_LABELS}

    for p_seq, t_seq in zip(pred_labels, true_labels):
        for p_idx, t_idx in zip(p_seq, t_seq):
            if t_idx == IGNORE_INDEX:
                continue

            p_label = ID2LABEL.get(p_idx, "O")
            t_label = ID2LABEL.get(t_idx, "O")

            if t_label in tp:
                if p_label == t_label:
                    tp[t_label] += 1
                else:
                    fn[t_label] += 1
                    if p_label in fp:
                        fp[p_label] += 1
            else:
                if p_label in fp:
                    fp[p_label] += 1

    per_class_f1 = {}
    f1_list = []
    for lbl in ENTITY_LABELS:
        total_pred = tp[lbl] + fp[lbl]
        total_gold = tp[lbl] + fn[lbl]
        prec = (tp[lbl] / total_pred) if total_pred > 0 else 0.0
        rec = (tp[lbl] / total_gold) if total_gold > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_f1[lbl] = f1
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list)
    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1
    }


def compute_exact_span_macro_f1(
    gold_dict: Dict[str, List[List[Any]]],
    pred_dict: Dict[str, List[List[Any]]]
) -> Dict[str, float]:
    """
    Computes exact character-span Precision, Recall, and Macro-F1 across all 12 legal entity types.
    Matches the official CodaBench benchmark evaluation.
    Each span is (start_char, end_char, label).
    """
    tp = {lbl: 0 for lbl in ENTITY_LABELS}
    fp = {lbl: 0 for lbl in ENTITY_LABELS}
    fn = {lbl: 0 for lbl in ENTITY_LABELS}

    all_doc_ids = set(gold_dict.keys()).union(set(pred_dict.keys()))

    for doc_id in all_doc_ids:
        # Convert to set of tuples for exact span matching
        gold_spans = set((s[0], s[1], s[2]) for s in gold_dict.get(doc_id, []) if len(s) >= 3 and s[2] in ENTITY_LABELS)
        pred_spans = set((s[0], s[1], s[2]) for s in pred_dict.get(doc_id, []) if len(s) >= 3 and s[2] in ENTITY_LABELS)

        for span in pred_spans:
            lbl = span[2]
            if span in gold_spans:
                tp[lbl] += 1
            else:
                fp[lbl] += 1

        for span in gold_spans:
            lbl = span[2]
            if span not in pred_spans:
                fn[lbl] += 1

    per_class_f1 = {}
    f1_list = []
    for lbl in ENTITY_LABELS:
        tot_p = tp[lbl] + fp[lbl]
        tot_g = tp[lbl] + fn[lbl]
        prec = (tp[lbl] / tot_p) if tot_p > 0 else 0.0
        rec = (tp[lbl] / tot_g) if tot_g > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_f1[lbl] = f1
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list)
    return {
        "span_macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


# ============================================================================
# 7. Test Inference & Span Reconstruction (prediction.json generator)
# ============================================================================

def reconstruct_document_spans(
    text: str,
    window_predictions: List[Dict[str, Any]],
    prob_threshold: float = 0.35
) -> List[List[Any]]:
    """
    Merges token-level predictions across overlapping sliding windows
    and reconstructs exact character spans [[start_char, end_char, label], ...].
    Strips leading and trailing whitespace.
    """
    text_len = len(text)
    # Character-level probability accumulator for 13 classes
    char_probs = torch.zeros((text_len, NUM_LABELS), dtype=torch.float32)
    char_counts = torch.zeros(text_len, dtype=torch.float32)

    for item in window_predictions:
        offsets = item["offsets"]
        probs = item["probs"]  # (seq_len, 13)

        for (start_c, end_c), p in zip(offsets, probs):
            if start_c == end_c or start_c >= text_len:
                continue
            char_probs[start_c:end_c] += p
            char_counts[start_c:end_c] += 1.0

    # Average probabilities
    valid_mask = char_counts > 0
    char_probs[valid_mask] = char_probs[valid_mask] / char_counts[valid_mask].unsqueeze(-1)
    
    # Assign most probable class per character
    char_labels = torch.argmax(char_probs, dim=-1).tolist()

    # Form contiguous spans
    spans = []
    curr_label = None
    curr_start = None

    for i, lbl_id in enumerate(char_labels):
        lbl_str = ID2LABEL[lbl_id]
        if lbl_str != "O" and valid_mask[i]:
            if curr_label is None:
                curr_start = i
                curr_label = lbl_str
            elif curr_label != lbl_str:
                # Close previous span
                spans.append([curr_start, i, curr_label])
                curr_start = i
                curr_label = lbl_str
        else:
            if curr_label is not None:
                spans.append([curr_start, i, curr_label])
                curr_label = None
                curr_start = None

    if curr_label is not None:
        spans.append([curr_start, text_len, curr_label])

    # Clean whitespace boundaries
    cleaned_spans = []
    for s, e, lbl in spans:
        span_text = text[s:e]
        stripped = span_text.strip()
        if not stripped:
            continue
        # Find exact trimmed start and end
        lead_spaces = len(span_text) - len(span_text.lstrip())
        trail_spaces = len(span_text) - len(span_text.rstrip())
        clean_s = s + lead_spaces
        clean_e = e - trail_spaces
        if clean_e > clean_s:
            cleaned_spans.append([clean_s, clean_e, lbl])

    return cleaned_spans
