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
    
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            local_feats: (B, N, D)
        
        Returns:
            selected_feats: (B, K, D)
            aux_loss: dict with diversity loss
        """
        B, N, D = local_feats.shape
        device = local_feats.device
        
        # Get scores from each head: [(B, N, 1), ...]
        head_scores = [scorer(local_feats) for scorer in self.scorers]  # num_heads × (B, N, 1)
        head_scores = torch.cat(head_scores, dim=-1)  # (B, N, num_heads)
        
        # Top-K per head
        k_per_head = max(1, self.num_select // self.num_heads)
        selected_indices_per_head = []
        
        for h in range(self.num_heads):
            scores_h = head_scores[:, :, h]  # (B, N)
            _, indices_h = torch.topk(scores_h, k=k_per_head, dim=1)  # (B, k_per_head)
            selected_indices_per_head.append(indices_h)
        
        # Union of indices to ensure diversity
        all_indices = torch.cat(selected_indices_per_head, dim=1)  # (B, num_heads * k_per_head)
        unique_indices = torch.unique(all_indices, dim=1)  # (B, K) where K <= num_heads * k_per_head
        
        # Pad or trim to exact K
        current_k = unique_indices.shape[1]
        if current_k < self.num_select:
            # Pad with remaining indices
            mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            mask.scatter_(1, unique_indices, True)
            remaining = (~mask).nonzero(as_tuple=True)
            padded_indices = torch.zeros(B, self.num_select - current_k, dtype=torch.long, device=device)
            for b in range(B):
                remaining_b = remaining[1][remaining[0] == b]
                if len(remaining_b) > 0:
                    padded_indices[b] = remaining_b[:self.num_select - current_k]
            unique_indices = torch.cat([unique_indices, padded_indices], dim=1)
        elif current_k > self.num_select:
            unique_indices = unique_indices[:, :self.num_select]
        
        # Gather selected features
        selected_feats = torch.gather(
            local_feats, 1,
            unique_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, K, D)
        
        # Diversity loss: encourage different heads to select different patches
        diversity_loss = self._compute_diversity_loss(head_scores, selected_indices_per_head)
        
        aux_loss = {'diversity': diversity_loss}
        return selected_feats, aux_loss
    
    def _compute_diversity_loss(self, head_scores: torch.Tensor, 
                               selected_indices_per_head: list) -> torch.Tensor:
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
    
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            local_feats: (B, N, D)
        
        Returns:
            slot_feats: (B, K, D) weighted slot features
            aux_loss: dict with orthogonality loss
        """
        B, N, D = local_feats.shape
        device = local_feats.device
        
        # Broadcast slots
        slots = self.slots.expand(B, -1, -1)  # (B, num_slots, D)
        
        # Top-K masking: select top-K patches based on importance
        importance = (local_feats ** 2).sum(dim=-1)  # (B, N)
        _, top_k_indices = torch.topk(importance, k=self.num_select, dim=1)  # (B, K)
        
        # Create mask for top-K patches
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, top_k_indices, True)
        
        # Apply sparse attention only to top-K patches
        local_feats_sparse = local_feats[mask].reshape(B, self.num_select, D)  # (B, K, D)
        
        # Slot attention: update slots via cross-attention
        slots_norm = self.norm1(slots)
        slots_updated, _ = self.mha(slots_norm, local_feats_sparse, local_feats_sparse)
        slots = slots + slots_updated
        
        # Feed-forward
        slots_ff = self.ff(self.norm2(slots))
        slots = slots + slots_ff
        
        # Compute slot weights (soft assignment)
        slot_logits = torch.bmm(slots, local_feats_sparse.transpose(1, 2))  # (B, num_slots, K)
        slot_weights = F.softmax(slot_logits, dim=2)  # (B, num_slots, K)
        
        # Weighted sum of patches per slot
        slot_feats = torch.bmm(slot_weights, local_feats_sparse)  # (B, num_slots, D)
        
        # Orthogonality loss: encourage different slots to focus on different regions
        ortho_loss = self._compute_orthogonality_loss(slots)
        
        aux_loss = {'orthogonality': ortho_loss}
        return slot_feats, aux_loss
    
    def _compute_orthogonality_loss(self, slots: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between slots (force orthogonality)."""
        # Normalize slots
        slots_norm = F.normalize(slots, dim=-1)  # (B, num_slots, D)
        
        # Pairwise cosine similarity
        sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # (B, num_slots, num_slots)
        
        # Loss = sum of off-diagonal similarities
        B, num_slots = sim.shape[0], sim.shape[1]
        ortho_loss = 0.0
        for b in range(B):
            sim_b = sim[b]
            sim_b.fill_diagonal_(0)
            ortho_loss += sim_b.abs().mean()
        ortho_loss = ortho_loss / B
        
        return ortho_loss


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
                text_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Use text features as query, selected features as key/value.
        
        Args:
            selected_feats: (B, K, D)
            text_feats: (B, D) query
        
        Returns:
            final_feats: (B, D)
        """
        if text_feats is None:
            return selected_feats.mean(dim=1)
        
        B, K, D = selected_feats.shape
        
        # Reshape text features as single query token
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
                                   margin: float = 0.1) -> torch.Tensor:
    """
    Margin ranking loss: final_feat should be closer to positive text 
    than to negative texts.
    
    Args:
        final_feat: (B, D)
        pos_text_feat: (B, D)
        neg_text_feats: (B, num_neg, D)
        margin: margin for ranking loss
    
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
    
    # Margin ranking: pos_sim should be > neg_sim + margin
    # Loss = max(0, neg_sim - pos_sim + margin)
    neg_max = neg_sims.max(dim=1)[0]  # (B,)
    loss = F.relu(neg_max - pos_sim + margin).mean()
    
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
        
        if selector_type == 'mlp':
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
        margin = 0.1
        neg_max = neg_sims.max(dim=1)[0]  # (B,)
        loss = F.relu(neg_max - pos_sim + margin).mean()
        
        return loss
    
    def compute_mixup_invariance_loss(self, selected_feats: torch.Tensor, 
                                     bg_mask: Optional[torch.Tensor] = None,
                                     labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Causal Mixup Invariance Loss: mixed features should maintain class predictions.
        
        Logic: mixed_feat = selected_feat + (1-bg_mask) * shuffled_bg_feat
        
        Args:
            selected_feats: (B, K, D) selected patch features
            bg_mask: (B, K) or (B,) background mask for selective mixing
            labels: (B,) optional ground truth labels
        
        Returns:
            loss: scalar
        """
        B, K, D = selected_feats.shape
        device = selected_feats.device
        
        # Average selected features as foreground
        fg_feat = selected_feats.mean(dim=1)  # (B, D)
        
        # Create a random background feature by shuffling
        shuffle_idx = torch.randperm(B, device=device)
        # Detach background features to prevent gradients flowing into background sources.
        # This enforces the causal intervention: selector must learn to extract foreground,
        # not adapt foreground to background changes.
        bg_feat = fg_feat[shuffle_idx].detach()
        
        # Create background mask if not provided
        if bg_mask is None:
            bg_mask = torch.ones(B, 1, device=device)
        elif bg_mask.dim() == 1:
            bg_mask = bg_mask.unsqueeze(1)  # (B, 1)
        
        # Causal mixup: mixed_feat = fg_feat + (1-bg_mask) * shuffled_bg_feat
        mixed_feat = fg_feat + (1.0 - bg_mask) * bg_feat  # (B, D)
        
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
                selected_feats: (B, K, D)
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
        selected_feats, sel_aux_loss = self.selector(local_features)  # (B, K, D), dict
        
        # Get text features for guided fusion
        text_feats = self._text_features.type(self.dtype).to(device)  # (num_classes, D)
        
        if labels is not None:
            pos_text_feat = text_feats[labels]  # (B, D)
        else:
            pos_text_feat = text_feats.mean(dim=0, keepdim=True).expand(B, -1)  # (B, D)
        
        # Feature fusion
        final_feats = self.fuser(selected_feats, pos_text_feat)  # (B, D)
        
        # Normalize
        final_feats = final_feats / (final_feats.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * final_feats @ text_feats.T  # (B, num_classes)
        
        # Collect auxiliary losses
        aux_losses = sel_aux_loss.copy()
        
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

                sem_excl_loss = compute_semantic_exclusion_loss(final_feats, pos_text_feat, neg_tensor)
                aux_losses['semantic_exclusion'] = sem_excl_loss
            else:
                neg_indices = torch.arange(self.num_classes, device=device).unsqueeze(0)
                neg_indices = neg_indices.expand(B, -1)
                # Mask out positive class
                mask = torch.ones_like(neg_indices, dtype=torch.bool)
                mask.scatter_(1, labels.unsqueeze(1), False)
                neg_text_feats = text_feats[mask].reshape(B, -1, self.feat_dim)
                sem_excl_loss = compute_semantic_exclusion_loss(final_feats, pos_text_feat, neg_text_feats)
                aux_losses['semantic_exclusion'] = sem_excl_loss
        
        # Causal Mixup Invariance Loss (if enabled)
        if self.cfg.get('use_mixup_invariance', False):
            mixup_loss = self.compute_mixup_invariance_loss(selected_feats, labels=labels)
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
    clip_model, _ = clip_module.load("ViT-B/32", device=cfg['device'])
    
    model = build_modular_model(cfg, classnames, clip_model)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(cfg['device'])
    labels = torch.tensor([0, 1]).to(cfg['device'])
    
    output = model(x, labels)
    print(f"✓ Forward pass successful")
    print(f"  logits shape: {output['logits'].shape}")
    print(f"  aux_losses: {output['aux_losses'].keys()}")
