"""
Modular OOD Detection Framework
Supports flexible feature selection, fusion, and loss combinations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import math
import clip
import json


# ============================================================================
# PART 1: BASE CLASSES
# ============================================================================

class BaseSelector(ABC, nn.Module):
    """Abstract base class for feature selectors."""
    
    def __init__(self, input_dim: int, num_select: int, cfg: Dict = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_select = num_select
        self.cfg = cfg or {}
    
    @abstractmethod
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            local_feats: (B, N, D) local patch features
        
        Returns:
            selected_feats: (B, K, D) selected features
            aux_loss: dict with auxiliary losses
        """
        pass


class BaseFuser(ABC, nn.Module):
    """Abstract base class for feature fusers."""
    
    def __init__(self, input_dim: int, cfg: Dict = None):
        super().__init__()
        self.input_dim = input_dim
        self.cfg = cfg or {}
    
    @abstractmethod
    def forward(self, selected_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            selected_feats: (B, K, D) selected features
            text_feats: (B, D) optional text features for guided fusion
        
        Returns:
            final_feats: (B, D) global representation
        """
        pass


# ============================================================================
# PART 2: FEATURE SELECTORS
# ============================================================================

class IdentitySelector(BaseSelector):
    """No-op selector that returns all features as-is (for baseline methods)."""
    
    def __init__(self, input_dim: int, num_select: int, cfg: Dict = None):
        super().__init__(input_dim, num_select, cfg)
    
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Args:
            local_feats: (B, N, D)
        
        Returns:
            selected_feats: (B, N, D) - all features, unchanged
            aux_loss: empty dict
            mask: (B, N, 1) - all ones mask since all features are considered foreground
        """
        B, N, D = local_feats.shape
        device = local_feats.device
        # Return all features without selection
        # For baseline, all features are considered foreground
        mask = torch.ones(B, N, 1, device=device)
        return local_feats, {}, mask


class MultiHeadMLPSelector(BaseSelector):
    """
    Multi-head MLP-based feature selector.
    Each head independently scores patches and selects top-K.
    Union of all selected indices ensures diversity.
    """
    
    def __init__(self, input_dim: int, num_select: int, num_heads: int = 4, cfg: Dict = None):
        super().__init__(input_dim, num_select, cfg)
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Each head scores patches independently
        self.scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, 1)
            ) for _ in range(num_heads)
        ])
        
        # Learnable head weights
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
    
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Args:
            local_feats: (B, N, D)
        
        Returns:
            selected_feats: (B, N, D) - features with STE mask applied
            aux_loss: dict with diversity loss
            ste_mask: (B, N, 1) - STE mask for Mixup
        """
        B, N, D = local_feats.shape
        device = local_feats.device
        
        # 1. 打分: 计算 scores (B, N, H)
        head_scores = [scorer(local_feats) for scorer in self.scorers]  # num_heads × (B, N, 1)
        scores = torch.cat(head_scores, dim=-1)  # (B, N, num_heads)
        
        # 2. 生成硬 Mask
        final_mask = torch.zeros(B, N, 1, device=device)
        k_per_head = max(1, self.num_select // self.num_heads)
        
        for h in range(self.num_heads):
            scores_h = scores[:, :, h]  # (B, N)
            _, topk_indices_h = torch.topk(scores_h, k=k_per_head, dim=1)  # (B, k_per_head)
            # 将这些位置在final_mask中置为1.0 (取并集)
            for b in range(B):
                final_mask[b, topk_indices_h[b], 0] = 1.0
        
        # 3. 应用 STE (Straight-Through Estimator)
        scores_max = scores.max(dim=-1)[0].unsqueeze(-1)  # (B, N, 1)
        ste_mask = (final_mask - scores_max).detach() + scores_max  # (B, N, 1)
        
        # 4. 应用 mask 到特征
        selected_feats = local_feats * ste_mask  # (B, N, D)
        
        # Diversity loss: encourage different heads to select different patches
        diversity_loss = self._compute_diversity_loss(scores, None)
        
        aux_loss = {'diversity': diversity_loss}
        return selected_feats, aux_loss, ste_mask
    
    def _compute_diversity_loss(self, head_scores: torch.Tensor, 
                               selected_indices_per_head: list = None) -> torch.Tensor:
        """Correlation loss between head score maps to encourage diversity."""
        # Flatten head scores: (B, N, num_heads) -> (B*N, num_heads)
        B, N, _ = head_scores.shape
        head_scores_flat = head_scores.reshape(B * N, self.num_heads)
        
        # Compute pairwise correlations
        head_scores_norm = F.normalize(head_scores_flat, dim=0)
        corr_matrix = torch.mm(head_scores_norm.t(), head_scores_norm)  # (num_heads, num_heads)
        
        # Zero out diagonal and take mean of off-diagonal
        corr_matrix.fill_diagonal_(0)
        diversity_loss = corr_matrix.abs().mean()
        
        return diversity_loss


class SparseSlotAttentionSelector(BaseSelector):
    """
    Slot Attention-based feature selector.
    Learnable slots attend to patches with top-K masking for sparsity.
    Slots are initialized with orthogonal initialization for better convergence.
    """
    
    def __init__(self, input_dim: int, num_select: int, num_slots: int = 4, cfg: Dict = None):
        super().__init__(input_dim, num_select, cfg)
        self.num_slots = num_slots
        
        # Learnable slot queries with orthogonal initialization
        # For orthogonal init, we need at least 2D tensor: (num_slots, input_dim)
        self.slots = nn.Parameter(torch.empty(1, num_slots, input_dim))
        nn.init.orthogonal_(self.slots.data.squeeze(0), gain=1.0)
        self.slots.data = self.slots.data.unsqueeze(0)
        
        # Cross-attention components
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.mha = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, input_dim)
        )
        
        # Importance scorer: lightweight projection layer for better patch selection
        self.importance_scorer = nn.Linear(input_dim, 1)
    
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Args:
            local_feats: (B, N, D)
        
        Returns:
            slot_feats: (B, num_slots, D) weighted slot features
            aux_loss: dict with orthogonality loss
            img_space_mask: (B, N, 1) - all Slots mask的并集,用于告诉Mixup哪些是前景
        """
        B, N, D = local_feats.shape
        device = local_feats.device
        
        # 1. 计算 attention logits
        # Broadcast slots
        slots = self.slots.expand(B, -1, -1)  # (B, num_slots, D)
        
        # 计算每个 slot 对每个 patch 的 attention
        local_feats_norm = F.normalize(local_feats, dim=-1)  # (B, N, D)
        slots_norm = F.normalize(slots, dim=-1)  # (B, num_slots, D)
        attn_logits = torch.bmm(slots_norm, local_feats_norm.transpose(1, 2))  # (B, num_slots, N)
        
        # 2. 生成硬 Mask
        # 找到 Top-K 的阈值
        k = self.num_select
        threshold, _ = torch.topk(attn_logits, k=k, dim=-1, sorted=True)
        threshold = threshold[:, :, -1:].expand(-1, -1, N)  # (B, num_slots, N)
        
        # 生成硬 mask
        mask_hard = (attn_logits >= threshold).float()  # (B, num_slots, N)
        
        # 3. 应用 STE
        mask = (mask_hard - attn_logits).detach() + attn_logits  # (B, num_slots, N)
        
        # 4. 加权: 计算注意力权重
        attn = F.softmax(attn_logits * mask - 1e9 * (1 - mask), dim=-1)  # (B, num_slots, N)
        
        # 计算 slot features
        slot_feats = torch.bmm(attn, local_feats)  # (B, num_slots, D)
        
        # 更新 slots
        slots_updated, _ = self.mha(self.norm1(slots), slot_feats, slot_feats)
        slots = slots + slots_updated
        
        slots_ff = self.ff(self.norm2(slots))
        slots = slots + slots_ff
        
        # 5. 生成 img_space_mask (所有 Slots mask 的并集)
        img_space_mask = mask_hard.max(dim=1)[0].unsqueeze(-1)  # (B, N, 1)
        
        # Orthogonality loss: encourage different slots to focus on different regions
        # Use slot_weights (attention maps) instead of slots themselves
        ortho_loss = self._compute_orthogonality_loss(attn)
        
        aux_loss = {'orthogonality': ortho_loss}
        return slots, aux_loss, img_space_mask
    
    def _compute_orthogonality_loss(self, slot_weights: torch.Tensor) -> torch.Tensor:
        """Orthogonality loss on attention maps to prevent overlapping attention."""
        # slot_weights shape: (B, num_slots, K)
        B, num_slots, K = slot_weights.shape
        
        # Compute Gram matrix for each sample in batch
        gram_list = []
        for b in range(B):
            # For each sample, compute gram matrix of shape (num_slots, num_slots)
            weights_b = slot_weights[b]  # (num_slots, K)
            gram_b = torch.mm(weights_b, weights_b.T)  # (num_slots, num_slots)
            gram_list.append(gram_b)
        
        # Stack to get (B, num_slots, num_slots)
        gram = torch.stack(gram_list, dim=0)
        
        # Create identity matrix for comparison
        I = torch.eye(num_slots, device=slot_weights.device).unsqueeze(0).expand(B, -1, -1)
        
        # Penalize non-diagonal elements
        loss = (gram - I).abs().mean()
        
        return loss


# ============================================================================
# PART 3: FEATURE FUSERS
# ============================================================================

class MeanPoolFuser(BaseFuser):
    """Simple mean pooling of selected features."""
    
    def forward(self, selected_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            selected_feats: (B, K, D)
            text_feats: unused
        
        Returns:
            final_feats: (B, D)
        """
        return selected_feats.mean(dim=1)


class QueryGuidedAttentionFuser(BaseFuser):
    """Attention-based fusion guided by text features."""
    
    def __init__(self, input_dim: int, num_heads: int = 4, cfg: Dict = None):
        super().__init__(input_dim, cfg)
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            input_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, selected_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Use text features as query, selected features as key/value.
        
        Args:
            selected_feats: (B, K, D)
            text_feats: (num_classes, D) text features for all classes or (B, D) query
            labels: (B,) optional ground truth labels for training
        
        Returns:
            final_feats: (B, D)
        """
        if text_feats is None:
            return selected_feats.mean(dim=1)
        
        B, K, D = selected_feats.shape
        
        # 检查text_feats的形状，如果是(num_classes, D)，则需要根据训练/推理情况处理
        if text_feats.dim() == 2 and text_feats.shape[0] != B:
            # text_feats是(num_classes, D)形状
            if self.training and labels is not None:
                # 训练时使用标签指导的文本特征作为Query
                query = text_feats[labels].unsqueeze(1)  # (B, 1, D)
            else:
                # 推理时或无Label时，使用所有ID文本特征的平均值作为通用Query
                generic_query = text_feats.mean(dim=0, keepdim=True)  # (1, D)
                query = generic_query.expand(B, 1, -1)  # (B, 1, D)
        else:
            # text_feats已经是(B, D)形状的查询
            query = text_feats.unsqueeze(1)  # (B, 1, D)
        
        # Attention: query=text, key/value=selected_feats
        attn_out, _ = self.mha(query, selected_feats, selected_feats)  # (B, 1, D)
        
        # Residual connection and output
        final_feats = self.norm(query + attn_out).squeeze(1)  # (B, D)
        
        return final_feats


class SelfAttentionFuser(BaseFuser):
    """Self-attention based fusion to model relationships between selected patches."""
    
    def __init__(self, input_dim: int, num_heads: int = 4, cfg: Dict = None):
        super().__init__(input_dim, cfg)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=4 * input_dim,
            batch_first=True,
            activation='gelu'
        )
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, selected_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply transformer self-attention then mean pool.
        
        Args:
            selected_feats: (B, K, D)
            text_feats: unused
        
        Returns:
            final_feats: (B, D)
        """
        # Self-attention over selected features
        attended = self.encoder_layer(selected_feats)  # (B, K, D)
        
        # Normalization
        attended = self.norm(attended)
        
        # Mean pooling
        final_feats = attended.mean(dim=1)  # (B, D)
        
        return final_feats


# ============================================================================
# PART 4: LOSS FUNCTIONS
# ============================================================================

def compute_diversity_loss(selector_type: str, selector) -> torch.Tensor:
    """Extract diversity loss from selector if available."""
    if hasattr(selector, 'diversity_loss'):
        return selector.diversity_loss
    return torch.tensor(0.0, device=next(selector.parameters()).device)


def compute_semantic_exclusion_loss(final_feat: torch.Tensor, 
                                   pos_text_feat: torch.Tensor,
                                   neg_text_feats: torch.Tensor,
                                   margin: float = 0.1, temperature: float = 1.0) -> torch.Tensor:
    """
    Margin ranking loss: final_feat should be closer to positive text 
    than to negative texts.
    
    Args:
        final_feat: (B, D)
        pos_text_feat: (B, D)
        neg_text_feats: (B, num_neg, D)
        margin: margin for ranking loss
        temperature: temperature for logsumexp softmax-like operation
    
    Returns:
        loss: scalar
    """
    # Positive similarity
    pos_sim = F.cosine_similarity(final_feat, pos_text_feat, dim=1)  # (B,)
    
    # Negative similarities
    B, num_neg, D = neg_text_feats.shape
    neg_sims = []
    for i in range(num_neg):
        neg_sim_i = F.cosine_similarity(final_feat, neg_text_feats[:, i], dim=1)  # (B,)
        neg_sims.append(neg_sim_i)
    neg_sims = torch.stack(neg_sims, dim=1)  # (B, num_neg)
    
    # 使用torch.logsumexp替代max()，增强损失函数稳定性
    neg_score = torch.logsumexp(neg_sims * temperature, dim=1) / temperature  # (B,)
    
    # Margin ranking: pos_sim should be > neg_score + margin
    loss = F.relu(neg_score - pos_sim + margin).mean()
    
    return loss


def compute_redundancy_loss(selected_feats: torch.Tensor) -> torch.Tensor:
    """
    Redundancy loss: penalize non-diagonal elements in the Gram matrix of selected features.
    Encourages diversity among selected features.
    
    Args:
        selected_feats: (B, K, D) selected patch features
    
    Returns:
        loss: scalar
    """
    B, K, D = selected_feats.shape
    device = selected_feats.device
    
    # Normalize features
    selected_feats_norm = selected_feats / (selected_feats.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Compute Gram matrix for each sample
    gram_list = []
    for b in range(B):
        feats_b = selected_feats_norm[b]  # (K, D)
        gram_b = torch.mm(feats_b, feats_b.T)  # (K, K)
        gram_list.append(gram_b)
    
    gram = torch.stack(gram_list, dim=0)  # (B, K, K)
    
    # Create mask to zero out diagonal elements
    diag_mask = torch.eye(K, device=device).unsqueeze(0).expand(B, -1, -1)
    off_diag = gram * (1 - diag_mask)
    
    # Penalize non-diagonal elements
    loss = off_diag.abs().mean()
    
    return loss


def compute_mixup_invariance_loss(final_feat: torch.Tensor,
                                 bg_feat: torch.Tensor,
                                 bg_mask: torch.Tensor,
                                 labels: torch.Tensor,
                                 text_feats: torch.Tensor,
                                 alpha: float = 0.2) -> torch.Tensor:
    """
    Causal Mixup Invariance Loss: enforces that mixed features maintain label consistency.
    
    Logic: mixed_feat = selected_feat + (1-bg_mask) * shuffled_bg_feat
    
    Args:
        final_feat: (B, D) selected foreground features
        bg_feat: (B, D) background features
        bg_mask: (B, 1) or (B, D) background mask for selective mixing
        labels: (B,) ground truth labels
        text_feats: (num_classes, D) text features for logit computation
        alpha: mixup parameter
    
    Returns:
        loss: scalar
    """
    B = final_feat.shape[0]
    device = final_feat.device
    
    # Ensure bg_mask is properly shaped
    if bg_mask.dim() == 1:
        bg_mask = bg_mask.unsqueeze(1)  # (B, 1)
    
    # Shuffle indices for background features
    shuffle_idx = torch.randperm(B, device=device)
    shuffled_bg = bg_feat[shuffle_idx]
    
    # Causal mixup: mixed_feat = selected_feat + (1-bg_mask) * shuffled_bg_feat
    mixed_feat = final_feat + (1.0 - bg_mask) * shuffled_bg
    
    # Normalize mixed features
    mixed_feat = mixed_feat / (mixed_feat.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Compute logits for mixed features
    logit_scale = 1.0 / 0.07  # Default CLIP temperature
    mixed_logits = logit_scale * mixed_feat @ text_feats.T  # (B, num_classes)
    
    # Mixed labels: labels[shuffle_idx] indicates which class the mixed features should belong to
    # We enforce that mixed features maintain the original label through contrastive loss
    # Loss: KL divergence between original logits and mixed logits (encouraging invariance)
    
    # Compute original logits
    orig_logits = logit_scale * final_feat @ text_feats.T  # (B, num_classes)
    
    # KL divergence loss: mixed logits should be similar to original logits
    orig_probs = F.softmax(orig_logits, dim=1)
    mixed_log_probs = F.log_softmax(mixed_logits, dim=1)
    
    mixup_loss = F.kl_div(mixed_log_probs, orig_probs, reduction='batchmean')
    
    return mixup_loss

# Note: The top-level function `compute_mixup_invariance_loss` was used earlier
# during development. The canonical implementation for the model is provided as
# `ModularCustomCLIP.compute_mixup_invariance_loss` (a class method) which
# includes the important causal `.detach()` behavior on background features.
# To avoid confusion and duplicated logic, this top-level helper is retained
# only as a thin wrapper that calls the class implementation when needed.
# If you prefer to remove this wrapper entirely, it's safe to delete it.


# ============================================================================
# PART 5: MAIN MODEL - CustomCLIP
# ============================================================================

class ModularCustomCLIP(nn.Module):
    """
    Modular CLIP-based OOD detection model.
    Supports flexible selector, fuser, and loss combinations.
    """
    
    def __init__(self, cfg: Dict, classnames: list, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.device = cfg.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.lab2idx = {c: i for i, c in enumerate(self.classnames)}
        # container for external OOD text features per ID class: {label_idx: (num_ood, D)}
        self.ood_text_feats = {}
        self.has_ood_map = False
        
        # CLIP components (frozen)
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # Get feature dimension
        try:
            self.feat_dim = self.image_encoder.output_dim
        except:
            self.feat_dim = clip_model.ln_final.weight.shape[0]
        
        # Initialize selector
        selector_type = cfg.get('selector_type', 'mlp')
        num_select = cfg.get('num_select', 49)
        
        if selector_type is None:
            # Baseline methods: no feature selection, use identity selector
            self.selector = IdentitySelector(self.feat_dim, num_select, cfg)
        elif selector_type == 'mlp':
            num_heads = cfg.get('num_heads_selector', 4)
            self.selector = MultiHeadMLPSelector(self.feat_dim, num_select, num_heads, cfg)
        elif selector_type == 'slot':
            num_slots = cfg.get('num_slots', 4)
            self.selector = SparseSlotAttentionSelector(self.feat_dim, num_select, num_slots, cfg)
        else:
            raise ValueError(f"Unknown selector_type: {selector_type}")
        
        # Initialize fuser
        fuser_type = cfg.get('fuser_type', 'mean')
        
        if fuser_type == 'mean':
            self.fuser = MeanPoolFuser(self.feat_dim, cfg)
        elif fuser_type == 'query_attn':
            num_heads = cfg.get('num_heads_fuser', 4)
            self.fuser = QueryGuidedAttentionFuser(self.feat_dim, num_heads, cfg)
        elif fuser_type == 'self_attn':
            num_heads = cfg.get('num_heads_fuser', 4)
            self.fuser = SelfAttentionFuser(self.feat_dim, num_heads, cfg)
        else:
            raise ValueError(f"Unknown fuser_type: {fuser_type}")
        
        # Get and cache text features
        self._cache_text_features()
        
        print(f"✓ ModularCustomCLIP initialized")
        print(f"  Selector: {selector_type} (num_select={num_select})")
        print(f"  Fuser: {fuser_type}")
    
    def _cache_text_features(self):
        """Pre-compute text features for all class names."""
        templates = self.cfg.get('templates', ["a photo of a"])
        if isinstance(templates, str):
            templates = [templates]
        
        text_features_list = []
        with torch.no_grad():
            for classname in self.classnames:
                classname = classname.replace('_', ' ')
                texts = [t.format(classname) for t in templates]
                texts = clip.tokenize(texts).to(self.device)
                text_embeddings = self.text_encoder.encode_text(texts)
                text_embeddings = text_embeddings / (text_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
                text_feat = text_embeddings.mean(dim=0)  # average across templates
                text_feat = text_feat / (text_feat.norm() + 1e-8)
                text_features_list.append(text_feat)
        
        self.text_features = torch.stack(text_features_list, dim=0).to(self.device).type(self.dtype)
        self.register_buffer('_text_features', self.text_features)

    def set_ood_classmap(self, ood_map: Dict[str, list], templates: Optional[list] = None):
        """
        Load external OOD classnames mapping and pre-compute their text features.

        Args:
            ood_map: dict mapping ID classname -> list of OOD classnames
            templates: optional list of templates for tokenization (defaults to cfg templates)
        """
        if templates is None:
            templates = self.cfg.get('templates', ["a photo of a {}"])
        if isinstance(templates, str):
            templates = [templates]

        self.ood_text_feats = {}
        with torch.no_grad():
            for id_name, ood_names in ood_map.items():
                # skip unknown id names
                if id_name not in self.lab2idx:
                    continue
                lab_idx = self.lab2idx[id_name]
                if not isinstance(ood_names, (list, tuple)) or len(ood_names) == 0:
                    continue

                texts = []
                for oname in ood_names:
                    oname = oname.replace('_', ' ')
                    for t in templates:
                        texts.append(t.format(oname))

                tokens = clip.tokenize(texts).to(self.device)
                t_emb = self.text_encoder.encode_text(tokens)
                t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-8)
                # Average across templates per OOD classname if templates >1
                num_per_ood = len(templates)
                if num_per_ood > 1:
                    t_emb = t_emb.view(-1, num_per_ood, t_emb.shape[-1]).mean(dim=1)

                # Normalize and store
                t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-8)
                self.ood_text_feats[lab_idx] = t_emb.to(self.device).type(self.dtype)

        self.has_ood_map = len(self.ood_text_feats) > 0
    
    def _compute_llm_negatives_loss(self, final_feats: torch.Tensor, 
                                    neg_text_tokens: torch.Tensor,
                                    pos_text_feats: torch.Tensor) -> torch.Tensor:
        """
        Compute LLM Negatives loss: push final features away from negative text tokens.
        
        Args:
            final_feats: (B, D) image features
            neg_text_tokens: (B, num_neg, D) negative text token features
            pos_text_feats: (B, D) positive text features
        
        Returns:
            loss: scalar
        """
        B, num_neg, D = neg_text_tokens.shape
        
        # Normalize features
        final_feats_norm = final_feats / (final_feats.norm(dim=-1, keepdim=True) + 1e-8)  # (B, D)
        pos_text_norm = pos_text_feats / (pos_text_feats.norm(dim=-1, keepdim=True) + 1e-8)  # (B, D)
        neg_text_norm = neg_text_tokens / (neg_text_tokens.norm(dim=-1, keepdim=True) + 1e-8)  # (B, num_neg, D)
        
        # Positive similarity (maximize)
        pos_sim = (final_feats_norm * pos_text_norm).sum(dim=1)  # (B,)
        
        # Negative similarities (minimize)
        neg_sims = torch.einsum('bd,bnd->bn', final_feats_norm, neg_text_norm)  # (B, num_neg)
        
        # Margin ranking loss: pos_sim should be > neg_sim + margin
        margin = self.cfg.get('margin', 0.1)
        neg_max = neg_sims.max(dim=1)[0]  # (B,)
        loss = F.relu(neg_max - pos_sim + margin).mean()
        
        return loss
    
    def compute_mixup_invariance_loss(self, selected_feats: torch.Tensor, 
                                     local_feats: torch.Tensor,
                                     bg_mask: torch.Tensor,
                                     labels: Optional[torch.Tensor] = None, 
                                     alpha: float = 0.2) -> torch.Tensor:
        """
        Causal Mixup Invariance Loss: mixed features should maintain class predictions.
        
        Logic: mixed_feat = fg_feat + alpha * shuffled_bg_feat
        
        Args:
            selected_feats: (B, N, D) selected patch features with STE mask applied
            local_feats: (B, N, D) original patch features
            bg_mask: (B, N, 1) - mask indicating foreground (1.0) and background (0.0)
            labels: (B,) optional ground truth labels
            alpha: mixup parameter
        
        Returns:
            loss: scalar
        """
        B, N, D = local_feats.shape
        device = local_feats.device
        
        # 1. 提取背景特征
        # bg_weights = (1 - bg_mask)  # (B, N, 1)
        # 计算背景特征的加权平均
        # bg_feat_pooled = (local_feats * bg_weights).sum(1) / (bg_weights.sum(1) + 1e-6)  # (B, D)
        
        # 计算前景特征的加权平均
        fg_feat = (local_feats * bg_mask).sum(1) / (bg_mask.sum(1) + 1e-6)  # (B, D)
        
        # 提取背景特征 (1 - bg_mask)，即非前景区域
        bg_weights = (1 - bg_mask)  # (B, N, 1)
        # 计算背景特征的加权平均
        bg_feat_pooled = (local_feats * bg_weights).sum(1) / (bg_weights.sum(1) + 1e-6)  # (B, D)
        
        # 关键：执行 detach() 防止背景特征参与梯度更新
        bg_feat_pooled = bg_feat_pooled.detach()
        
        # 2. Shuffle 背景索引
        shuffle_idx = torch.randperm(B, device=device)
        shuffled_bg = bg_feat_pooled[shuffle_idx]  # (B, D)
        
        # 3. 混合：mixed_feat = fg_feat + alpha * shuffled_bg_feat
        mixed_feat = fg_feat + alpha * shuffled_bg  # (B, D)
        
        # 4. 计算 Loss
        # Normalize
        fg_feat_norm = fg_feat / (fg_feat.norm(dim=-1, keepdim=True) + 1e-8)
        mixed_feat_norm = mixed_feat / (mixed_feat.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute logits for original and mixed features
        text_feats = self._text_features.type(self.dtype).to(device)
        logit_scale = self.logit_scale.exp()
        
        orig_logits = logit_scale * fg_feat_norm @ text_feats.T  # (B, num_classes)
        mixed_logits = logit_scale * mixed_feat_norm @ text_feats.T  # (B, num_classes)
        
        # KL divergence: mixed logits should be similar to original logits
        orig_probs = F.softmax(orig_logits, dim=1)
        mixed_log_probs = F.log_softmax(mixed_logits, dim=1)
        
        mixup_loss = F.kl_div(mixed_log_probs, orig_probs.detach(), reduction='batchmean')
        
        return mixup_loss
    
    def forward(self, image: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                negative_text_tokens: Optional[torch.Tensor] = None) -> Dict:
        """
        Args:
            image: (B, 3, H, W)
            labels: (B,) optional ground truth labels
            negative_text_tokens: (B, num_neg, D) optional LLM negative text tokens
        
        Returns:
            dict with:
                logits: (B, num_classes)
                aux_losses: dict of auxiliary losses
                selected_feats: (B, N, D) with mask applied
                final_feats: (B, D)
        """
        B = image.shape[0]
        device = image.device
        
        # Image encoding
        with torch.no_grad():
            image_features, local_features = self.image_encoder(image.type(self.dtype))
        
        # Normalize
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
        local_features = local_features / (local_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Feature selection
        selected_feats, sel_aux_loss, bg_mask = self.selector(local_features)  # (B, N, D), dict, (B, N, 1)
        
        # Get text features for guided fusion
        text_feats = self._text_features.type(self.dtype).to(device)  # (num_classes, D)
        
        if labels is not None:
            pos_text_feat = text_feats[labels]  # (B, D)
        else:
            pos_text_feat = text_feats.mean(dim=0, keepdim=True).expand(B, -1)  # (B, D)
        
        # Feature fusion
        final_feats = self.fuser(selected_feats, self.text_features, labels)  # (B, D)
        
        # Normalize
        final_feats = final_feats / (final_feats.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * final_feats @ text_feats.T  # (B, num_classes)
        
        # Collect auxiliary losses
        aux_losses = sel_aux_loss.copy()
        
        # Redundancy loss: encourage diversity among selected features
        if self.cfg.get('use_redundancy_loss', False):
            redundancy_loss = compute_redundancy_loss(selected_feats)
            lambda_redundancy = self.cfg.get('lambda_redundancy', 0.1)
            aux_losses['redundancy'] = lambda_redundancy * redundancy_loss
        
        # LLM Negatives Loss (if provided)
        if negative_text_tokens is not None and self.cfg.get('use_llm_negatives', False):
            llm_neg_loss = self._compute_llm_negatives_loss(final_feats, negative_text_tokens, pos_text_feat)
            lambda_llm = self.cfg.get('lambda_llm_negatives', 0.05)
            aux_losses['llm_negatives'] = lambda_llm * llm_neg_loss
        
        # Semantic exclusion loss (if enabled)
        if self.cfg.get('use_semantic_exclusion', False) and labels is not None:
            # If external OOD mapping provided, use precomputed ood_text_feats for each label
            if getattr(self, 'has_ood_map', False):
                # Build per-sample neg_text_feats padded to max_neg
                per_sample_feats = []
                max_neg = 0
                for b in range(B):
                    lab = int(labels[b].item())
                    if lab in self.ood_text_feats:
                        neg = self.ood_text_feats[lab]  # (num_ood, D)
                    else:
                        # fallback: all other ID text feats
                        mask = torch.ones(self.num_classes, dtype=torch.bool, device=device)
                        mask[lab] = False
                        neg = text_feats[mask]
                    per_sample_feats.append(neg)
                    if neg.shape[0] > max_neg:
                        max_neg = neg.shape[0]

                # Pad to max_neg
                neg_tensor = torch.zeros((B, max_neg, self.feat_dim), device=device, dtype=self.dtype)
                for b in range(B):
                    neg_b = per_sample_feats[b]
                    neg_tensor[b, :neg_b.shape[0], :] = neg_b

                margin = self.cfg.get('margin', 0.1)
                sem_excl_loss = compute_semantic_exclusion_loss(final_feats, pos_text_feat, neg_tensor, margin=margin)
                aux_losses['semantic_exclusion'] = sem_excl_loss
            else:
                neg_indices = torch.arange(self.num_classes, device=device).unsqueeze(0)
                neg_indices = neg_indices.expand(B, -1)
                # Mask out positive class
                mask = torch.ones_like(neg_indices, dtype=torch.bool)
                mask.scatter_(1, labels.unsqueeze(1), False)
                neg_text_feats = text_feats[mask].reshape(B, -1, self.feat_dim)
                margin = self.cfg.get('margin', 0.1)
                sem_excl_loss = compute_semantic_exclusion_loss(final_feats, pos_text_feat, neg_text_feats, margin=margin)
                aux_losses['semantic_exclusion'] = sem_excl_loss
        
        # Causal Mixup Invariance Loss (if enabled)
        if self.cfg.get('use_mixup_invariance', False):
            mixup_loss = self.compute_mixup_invariance_loss(selected_feats, local_features, bg_mask, labels=labels)
            lambda_mixup = self.cfg.get('lambda_mixup', 0.1)
            aux_losses['mixup_invariance'] = lambda_mixup * mixup_loss
        
        return {
            'logits': logits,
            'aux_losses': aux_losses,
            'selected_feats': selected_feats,
            'final_feats': final_feats,
            'text_feats': text_feats,
        }


# ============================================================================
# Helper function to build model
# ============================================================================

def build_modular_model(cfg: Dict, classnames: list, clip_model):
    """Factory function to create modular model."""
    return ModularCustomCLIP(cfg, classnames, clip_model)


if __name__ == '__main__':
    # Quick test
    import clip as clip_module
    
    cfg = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'selector_type': 'mlp',  # or 'slot'
        'fuser_type': 'query_attn',  # or 'mean', 'self_attn'
        'num_select': 49,
        'num_heads_selector': 4,
        'num_heads_fuser': 4,
        'templates': ["a photo of a"],
        'use_semantic_exclusion': True,
    }
    
    classnames = ["dog", "cat", "bird"]
    clip_model, _ = clip_module.load("ViT-B/16", device=cfg['device'])
    
    model = build_modular_model(cfg, classnames, clip_model)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(cfg['device'])
    labels = torch.tensor([0, 1]).to(cfg['device'])
    
    output = model(x, labels)
    print(f"✓ Forward pass successful")
    print(f"  logits shape: {output['logits'].shape}")
    print(f"  aux_losses: {output['aux_losses'].keys()}")
