"""
Modular OOD Detection Framework (Final Clean Version)
Supports flexible feature selection, fusion, and loss combinations.
Optimized for Automatic Mixed Precision (AMP) with @autocast decorators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import math
import clip
import json
from torch.cuda.amp import autocast


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
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Returns:
            selected_feats: (B, N, D) or (B, K, D)
            aux_loss: dict
            mask: (B, N, 1) binary-like mask for mixup
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
                text_feats: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


# ============================================================================
# PART 2: FEATURE SELECTORS (Precision Managed)
# ============================================================================

class IdentitySelector(BaseSelector):
    """No-op selector."""
    
    def __init__(self, input_dim: int, num_select: int, cfg: Dict = None):
        super().__init__(input_dim, num_select, cfg)
    
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        B, N, D = local_feats.shape
        # Return all ones mask
        mask = torch.ones(B, N, 1, device=local_feats.device, dtype=local_feats.dtype)
        return local_feats, {}, mask


class MultiHeadMLPSelector(BaseSelector):
    """
    Multi-head MLP-based feature selector with STE.
    Runs scorer in mixed precision, but selection logic in FP32.
    """
    
    def __init__(self, input_dim: int, num_select: int, num_heads: int = 4, cfg: Dict = None):
        super().__init__(input_dim, num_select, cfg)
        self.num_heads = num_heads
        
        self.scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, 1)
            ) for _ in range(num_heads)
        ])
    
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        B, N, D = local_feats.shape
        dtype = local_feats.dtype
        device = local_feats.device
        
        # 1. Scorer can run in FP16 (faster)
        head_scores_list = [scorer(local_feats) for scorer in self.scorers]
        scores = torch.cat(head_scores_list, dim=-1)  # (B, N, H)
        
        # 2. Selection Logic (Force FP32 for stability with TopK & Indices)
        with autocast(enabled=False):
            scores_fp32 = scores.float()
            final_mask = torch.zeros(B, N, 1, device=device, dtype=torch.float32)
            k_per_head = max(1, self.num_select // self.num_heads)
            
            for h in range(self.num_heads):
                scores_h = scores_fp32[:, :, h]
                _, topk_indices_h = torch.topk(scores_h, k=k_per_head, dim=1)
                
                # Vectorized scatter for efficiency
                # Create a src tensor of ones: (B, k, 1)
                src = torch.ones(B, k_per_head, 1, device=device, dtype=torch.float32)
                # Expand indices: (B, k, 1)
                indices = topk_indices_h.unsqueeze(-1)
                
                final_mask.scatter_(1, indices, src)
                
            final_mask = torch.clamp(final_mask, 0.0, 1.0)
            
            # 3. STE (Straight-Through Estimator)
            scores_max = scores_fp32.max(dim=-1)[0].unsqueeze(-1)
            ste_mask = (final_mask - scores_max).detach() + scores_max
            
            # Diversity Loss Calculation (FP32)
            diversity_loss = self._compute_diversity_loss_fp32(scores_fp32)

        # 4. Apply mask (Back to original dtype)
        ste_mask = ste_mask.to(dtype)
        selected_feats = local_feats * ste_mask
        
        return selected_feats, {'diversity': diversity_loss.to(dtype)}, ste_mask
    
    @staticmethod
    def _compute_diversity_loss_fp32(head_scores: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss in FP32."""
        B, N, num_heads = head_scores.shape
        # Normalize along patch dim
        head_scores_norm = F.normalize(head_scores, dim=1, eps=1e-6)
        # Gram matrix: (B, H, H)
        gram_matrix = torch.bmm(head_scores_norm.transpose(1, 2), head_scores_norm)
        # Identity
        I = torch.eye(num_heads, device=head_scores.device).unsqueeze(0).expand(B, -1, -1)
        return (gram_matrix - I).abs().mean()


class SparseSlotAttentionSelector(BaseSelector):
    """
    Slot Attention selector.
    Critically optimized: Logic runs in FP32 to prevent NaN during softmax/exp.
    """
    
    def __init__(self, input_dim: int, num_select: int, num_slots: int = 8, cfg: Dict = None):
        super().__init__(input_dim, num_select, cfg)
        self.num_slots = num_slots
        
        # Orthogonal init
        self.slots = nn.Parameter(torch.empty(num_slots, input_dim))
        nn.init.orthogonal_(self.slots.data, gain=1.0)
        self.slots.data = self.slots.data.unsqueeze(0) # (1, S, D)
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.mha = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, input_dim)
        )
        self.temperature = cfg.get('selector_temperature', 0.1) if cfg else 0.1

    @autocast(enabled=False)
    def _forward_fp32(self, local_feats: torch.Tensor, slots: torch.Tensor):
        """Internal FP32 logic for numerical stability."""
        B, N, D = local_feats.shape
        device = local_feats.device
        
        # 1. Cosine Similarity Logits
        x_norm = F.normalize(local_feats, dim=-1, eps=1e-6)
        s_norm = F.normalize(slots, dim=-1, eps=1e-6)
        
        # Scale by 10 for better gradient flow
        attn_logits = torch.einsum('bkd,bnd->bkn', s_norm, x_norm)

        # 2. Top-K Masking
        # Determine K per slot or global K
        patches_per_slot = self.cfg.get('patches_per_slot_attn', 
                                      max(1, self.num_select // self.num_slots))
        patches_per_slot = min(patches_per_slot, N)
        
        topk_val, _ = torch.topk(attn_logits, k=patches_per_slot, dim=-1)
        threshold = topk_val[:, :, -1].unsqueeze(-1) # (B, S, 1)
        
        mask_hard = (attn_logits >= threshold).float()

        # 3. STE
        # Use sigmoid to approximate gradient
        mask_soft = torch.sigmoid(attn_logits / self.temperature)
        mask = (mask_hard - mask_soft).detach() + mask_soft

        # 4. Masked Softmax (Key for stability)
        neg_inf = -1e4 # Safe value for FP32/FP16
        masked_logits = attn_logits * mask + neg_inf * (1.0 - mask)
        attn = F.softmax(masked_logits, dim=-1)

        # 5. Weighted Sum
        slot_feats = torch.einsum('bkn,bnd->bkd', attn, local_feats)
        
        # 6. Global Mask (Union over slots) for Mixup
        img_space_mask = mask.max(dim=1)[0].unsqueeze(-1) # (B, N, 1)
        
        # 7. Orthogonality Loss
        # Gram matrix of attention maps
        gram = torch.bmm(attn, attn.transpose(1, 2))
        I = torch.eye(self.num_slots, device=device).unsqueeze(0).expand(B, -1, -1)
        ortho_loss = (gram - I).abs().mean()
        
        return slot_feats, ortho_loss, img_space_mask

    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        B, N, D = local_feats.shape
        dtype = local_feats.dtype
        
        # Expand slots for batch
        slots_expanded = self.slots.expand(B, -1, -1)
        
        # Run heavy lifting in FP32
        slot_feats_fp32, ortho_loss, img_space_mask_fp32 = self._forward_fp32(
            local_feats.float(), slots_expanded.float()
        )
        
        # Slot Update (can run in original dtype)
        slot_feats = slot_feats_fp32.to(dtype)
        slots = slots_expanded.to(dtype)
        
        # Standard Transformer Update
        slots_updated, _ = self.mha(self.norm1(slots), slot_feats, slot_feats)
        # slots = slots + slots_updated
        # slots = slots + self.ff(self.norm2(slots))
        
        return slots, {'orthogonality': ortho_loss.to(dtype)}, img_space_mask_fp32.to(dtype)


# ============================================================================
# PART 3: FEATURE FUSERS
# ============================================================================

class MeanPoolFuser(BaseFuser):
    def forward(self, selected_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        return selected_feats.mean(dim=1)


class QueryGuidedAttentionFuser(BaseFuser):
    def __init__(self, input_dim: int, num_heads: int = 4, cfg: Dict = None):
        super().__init__(input_dim, cfg)
        self.mha = nn.MultiheadAttention(input_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, selected_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if text_feats is None:
            return selected_feats.mean(dim=1)
        
        B = selected_feats.shape[0]
        dtype = selected_feats.dtype
        
        # Determine query
        # text_feats can be (Num_Classes, D) or (B, D)
        if text_feats.dim() == 2 and text_feats.shape[0] != B:
            if self.training and labels is not None:
                query = text_feats[labels].unsqueeze(1) # (B, 1, D)
            else:
                generic_query = text_feats.mean(dim=0, keepdim=True)
                query = generic_query.expand(B, 1, -1)
        else:
            query = text_feats.unsqueeze(1)
            
        query = query.to(dtype)
        
        attn_out, _ = self.mha(query, selected_feats, selected_feats)
        final_feats = self.norm(query + attn_out).squeeze(1)
        return final_feats


# class SelfAttentionFuser(BaseFuser):
#     """Fixed: Added correct arguments to forward matching BaseFuser."""
#     def __init__(self, input_dim: int, num_heads: int = 4, cfg: Dict = None):
#         super().__init__(input_dim, cfg)
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim, nhead=num_heads, dim_feedforward=4*input_dim,
#             batch_first=True, activation='gelu'
#         )
#         self.norm = nn.LayerNorm(input_dim)
    
#     def forward(self, selected_feats: torch.Tensor, 
#                 text_feats: Optional[torch.Tensor] = None, 
#                 labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        
#         attended = self.encoder_layer(selected_feats)
#         attended = self.norm(attended)
#         return attended.mean(dim=1)


class SelfAttentionFuser(BaseFuser):
    def __init__(self, input_dim, num_heads=4, cfg=None):
        super().__init__(input_dim, cfg)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads,
            dim_feedforward=4*input_dim,
            batch_first=True, activation='gelu'
        )
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )
        nn.init.normal_(self.proj[-1].weight, std=1e-3)

    def forward(self, selected_feats, text_feats=None, labels=None):
        x = self.encoder_layer(selected_feats)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)



# ============================================================================
# PART 4: LOSS FUNCTIONS (FP32 Enforced)
# ============================================================================

def compute_diversity_loss(selector_type: str, selector) -> torch.Tensor:
    if hasattr(selector, 'aux_loss') and 'diversity' in selector.aux_loss:
        return selector.aux_loss['diversity']
    return torch.tensor(0.0, device=next(selector.parameters()).device)

@autocast(enabled=False)
def compute_semantic_exclusion_loss(final_feat: torch.Tensor, 
                                   pos_text_feat: torch.Tensor,
                                   neg_text_feats: torch.Tensor,
                                   margin: float = 0.1, 
                                   logit_scale: float = 100.0) -> torch.Tensor:
    """
    Robust Ranking Loss in FP32.
    """
    # 1. Cast inputs to Float
    final_feat = final_feat.float()
    pos_text_feat = pos_text_feat.float()
    neg_text_feats = neg_text_feats.float()

    # 2. Normalize
    final_feat = F.normalize(final_feat, dim=-1, eps=1e-8)
    pos_text_feat = F.normalize(pos_text_feat, dim=-1, eps=1e-8)
    neg_text_feats = F.normalize(neg_text_feats, dim=-1, eps=1e-8)

    # 3. Similarities & Scaling
    pos_sim = (final_feat * pos_text_feat).sum(dim=1) * logit_scale
    neg_sims = torch.einsum("bd,bnd->bn", final_feat, neg_text_feats) * logit_scale

    # 4. Stable LogSumExp
    neg_score = torch.logsumexp(neg_sims, dim=1)
    
    # 5. Margin
    scaled_margin = margin * logit_scale
    
    # 6. Loss
    loss = F.relu(neg_score - pos_sim + scaled_margin)
    
    return loss.mean()

@autocast(enabled=False)
def compute_redundancy_loss(selected_feats: torch.Tensor) -> torch.Tensor:
    """Robust Redundancy Loss in FP32."""
    selected_feats = selected_feats.float()
    B, K, D = selected_feats.shape
    
    selected_feats_norm = F.normalize(selected_feats, dim=-1, eps=1e-8)
    gram = torch.bmm(selected_feats_norm, selected_feats_norm.transpose(1, 2))
    
    diag_mask = torch.eye(K, device=selected_feats.device).unsqueeze(0).expand(B, -1, -1)
    off_diag = gram * (1 - diag_mask)
    
    return off_diag.abs().mean()

@autocast(enabled=False)
def compute_mixup_invariance_loss(final_feat: torch.Tensor,
                                 local_feats: torch.Tensor,
                                 bg_mask: torch.Tensor,
                                 text_feats: torch.Tensor,
                                 labels: Optional[torch.Tensor] = None,
                                 logit_scale: float = 100.0,
                                 alpha: float = 0.2) -> torch.Tensor:
    """Robust Mixup Loss in FP32."""
    final_feat = final_feat.float()
    local_feats = local_feats.float()
    bg_mask = bg_mask.float()
    text_feats = text_feats.float()
    
    B = final_feat.shape[0]
    
    # Foreground Feature
    bg_mask_sum = bg_mask.sum(1) + 1e-8
    fg_feat = (local_feats * bg_mask).sum(1) / bg_mask_sum
    
    # Background Feature
    bg_weights = (1.0 - bg_mask)
    bg_weights_sum = bg_weights.sum(1) + 1e-8
    bg_feat_pooled = (local_feats * bg_weights).sum(1) / bg_weights_sum
    
    # Causal Intervention: Detach Background
    bg_feat_pooled = bg_feat_pooled.detach()
    
    # Shuffle & Mix
    shuffle_idx = torch.randperm(B, device=final_feat.device)
    shuffled_bg = bg_feat_pooled[shuffle_idx]
    mixed_feat = fg_feat + alpha * shuffled_bg
    
    # Logits
    fg_feat_norm = F.normalize(fg_feat, dim=-1, eps=1e-8)
    mixed_feat_norm = F.normalize(mixed_feat, dim=-1, eps=1e-8)
    
    orig_logits = logit_scale * fg_feat_norm @ text_feats.T
    mixed_logits = logit_scale * mixed_feat_norm @ text_feats.T
    
    # Stable KL Divergence
    # Shift logits for stability
    orig_logits = orig_logits - orig_logits.max(dim=1, keepdim=True)[0].detach()
    mixed_logits = mixed_logits - mixed_logits.max(dim=1, keepdim=True)[0].detach()
    
    orig_probs = F.softmax(orig_logits, dim=1)
    mixed_log_probs = F.log_softmax(mixed_logits, dim=1)
    
    loss = F.kl_div(mixed_log_probs, orig_probs.detach(), reduction='batchmean')
    return loss


# ============================================================================
# PART 5: MAIN MODEL - CustomCLIP
# ============================================================================

class ModularCustomCLIP(nn.Module):
    def __init__(self, cfg: Dict, classnames: list, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.device = cfg.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.lab2idx = {c: i for i, c in enumerate(self.classnames)}
        self.ood_text_feats = {}
        self.has_ood_map = False
        
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # Freeze backbone
        for p in self.image_encoder.parameters(): p.requires_grad = False
        for p in self.text_encoder.parameters(): p.requires_grad = False
        
        try: self.feat_dim = self.image_encoder.output_dim
        except: self.feat_dim = clip_model.ln_final.weight.shape[0]
        
        self._build_components()
        self._cache_text_features()
        print(f"✓ ModularCustomCLIP initialized. Dtype: {self.dtype}")

    def _build_components(self):
        s_type = self.cfg.get('selector_type', 'mlp')
        num_sel = self.cfg.get('num_select', 49)
        
        if s_type == 'mlp':
            self.selector = MultiHeadMLPSelector(self.feat_dim, num_sel, self.cfg.get('num_heads_selector', 4), self.cfg)
        elif s_type == 'slot':
            self.selector = SparseSlotAttentionSelector(self.feat_dim, num_sel, self.cfg.get('num_slots', 8), self.cfg)
        else:
            self.selector = IdentitySelector(self.feat_dim, num_sel, self.cfg)
        # self.selector = self.selector.to(self.dtype)
        
        f_type = self.cfg.get('fuser_type', 'mean')
        if f_type == 'query_attn':
            self.fuser = QueryGuidedAttentionFuser(self.feat_dim, self.cfg.get('num_heads_fuser', 4), self.cfg)
        elif f_type == 'self_attn':
            self.fuser = SelfAttentionFuser(self.feat_dim, self.cfg.get('num_heads_fuser', 4), self.cfg)
        else:
            self.fuser = MeanPoolFuser(self.feat_dim, self.cfg)
        # self.fuser = self.fuser.to(self.dtype)

    def _cache_text_features(self):
        templates = self.cfg.get('templates', ["a photo of a {}"])
        if isinstance(templates, str): templates = [templates]
        
        text_features_list = []
        with torch.no_grad():
            for classname in self.classnames:
                classname = classname.replace('_', ' ')
                texts = [t.format(classname) for t in templates]
                texts = clip.tokenize(texts).to(self.device)
                text_embeddings = self.text_encoder.encode_text(texts)
                text_embeddings = text_embeddings / (text_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
                text_feat = text_embeddings.mean(dim=0)
                text_feat = text_feat / (text_feat.norm() + 1e-8)
                text_features_list.append(text_feat)
        
        self.text_features = torch.stack(text_features_list, dim=0).to(self.device).type(self.dtype)
        self.register_buffer('_text_features', self.text_features)

    def set_ood_classmap(self, ood_map: Dict[str, list], templates: Optional[list] = None):
        # ... (Same as before, skipped for brevity)
        pass

    def _compute_llm_negatives_loss_wrapper(self, final_feats, neg_text_tokens, pos_text_feats):
        # Wrapper to pass logit scale
        scale_val = self.logit_scale.exp().clamp(max=100.0).item()
        return compute_semantic_exclusion_loss(
            final_feats, pos_text_feats, neg_text_tokens,
            margin=self.cfg.get('margin', 0.1),
            logit_scale=scale_val
        )

    def forward(self, image: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                negative_text_tokens: Optional[torch.Tensor] = None) -> Dict:

        B = image.shape[0]
        
        # 1. Encode Image (FP16)
        image_features, local_features = self.image_encoder(image.type(self.dtype))

        
        image_features = F.normalize(image_features, dim=-1)
        local_features = F.normalize(local_features, dim=-1)
        
        # 2. Select Features (Mixed Precision managed internally)
        selected_feats, sel_aux_loss, bg_mask = self.selector(local_features)

        # 3. Text Features
        text_feats = self._text_features.type(self.dtype)
        if labels is not None:
            pos_text_feat = text_feats[labels]
        else:
            pos_text_feat = text_feats.mean(dim=0, keepdim=True).expand(B, -1)
            
        # 4. Fusion
        final_feats = self.fuser(selected_feats, text_feats, labels)
        # final_feats = F.normalize(image_features+final_feats, dim=-1)
        final_feats = F.normalize(image_features, dim=-1)

        # 5. Logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * final_feats @ text_feats.T
        
        # 6. Losses
        aux_losses = sel_aux_loss.copy()
        
        # A. Redundancy
        if self.cfg.get('use_redundancy_loss', False):
            aux_losses['redundancy'] = self.cfg.get('lambda_redundancy', 0.1) * \
                                     compute_redundancy_loss(selected_feats)
        
        # B. LLM Negatives
        if negative_text_tokens is not None and self.cfg.get('use_llm_negatives', False):
            scale_val = logit_scale.item()
            llm_loss = compute_semantic_exclusion_loss(
                final_feats, pos_text_feat, negative_text_tokens,
                margin=self.cfg.get('margin', 0.1),
                logit_scale=scale_val
            )
            aux_losses['llm_negatives'] = self.cfg.get('lambda_llm_negatives', 0.05) * llm_loss
            
        # C. Semantic Exclusion (All other classes)
        if self.cfg.get('use_semantic_exclusion', False) and labels is not None:
            mask = torch.ones(B, self.num_classes, dtype=torch.bool, device=self.device)
            mask[torch.arange(B), labels] = False
            neg_text_feats = text_feats.unsqueeze(0).expand(B, -1, -1)[mask].reshape(B, -1, self.feat_dim)
            
            scale_val = logit_scale.item()
            sem_loss = compute_semantic_exclusion_loss(
                final_feats, pos_text_feat, neg_text_feats,
                margin=self.cfg.get('margin', 0.1),
                logit_scale=scale_val
            )
            aux_losses['semantic_exclusion'] = sem_loss
            
        # D. Mixup
        if self.cfg.get('use_mixup_invariance', False):
            scale_val = logit_scale.item()
            mixup_loss = compute_mixup_invariance_loss(
                final_feats, local_features, bg_mask,
                text_feats=text_feats,
                labels=labels,
                logit_scale=scale_val,
                alpha=self.cfg.get('mixup_alpha', 0.2)
            )
            aux_losses['mixup_invariance'] = self.cfg.get('lambda_mixup', 0.1) * mixup_loss
            
        return {
            'logits': logits,
            'aux_losses': aux_losses,
            'selected_feats': selected_feats,
            'final_feats': final_feats
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