import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.distributions import Bernoulli

import clip


class SimpleTextEncoder(nn.Module):
    """Encode fixed text prompts using pre-trained CLIP text encoder."""
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def forward(self, classnames, templates=["a photo of a"]):
        """
        Encode class names with fixed templates.
        
        Args:
            classnames: list of class names (e.g., ["dog", "cat", "bird"])
            templates: list of template strings (e.g., ["a photo of a", "a picture of a"])
        
        Returns:
            text_features: [num_classes, embedding_dim]
        """
        text_features_list = []
        
        for classname in classnames:
            classname = classname.replace('_', ' ')
            # Create prompts by combining templates and class names
            texts = [t.format(classname) for t in templates]
            # Tokenize and encode
            texts_tokens = clip.tokenize(texts).to(next(self.clip_model.parameters()).device)
            with torch.no_grad():
                class_embeddings = self.clip_model.encode_text(texts_tokens)
                class_embeddings = class_embeddings / (class_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
                # Average across templates
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / (class_embedding.norm() + 1e-8)
            text_features_list.append(class_embedding)
        
        text_features = torch.stack(text_features_list, dim=0)
        return text_features


class GatingNetwork(nn.Module):
    """Per-token gating network for feature selection."""
    def __init__(self, local_dim, gate_hidden=256):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(local_dim * 2, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, local_features, text_features):
        """
        Args:
            local_features: [B, T, C] local image features
            text_features: [B, C] text features (broadcasted to [B, 1, C])
        
        Returns:
            gate_logits: [B, T] per-token gating logits
        """
        B, T, C = local_features.shape
        # Expand text features to match local features
        text_exp = text_features.unsqueeze(1).expand(-1, T, -1)  # [B, T, C]
        # Concatenate
        gate_input = torch.cat([local_features, text_exp], dim=-1)  # [B, T, 2C]
        gate_input_flat = gate_input.view(B * T, -1)
        gate_logits = self.gating(gate_input_flat).view(B, T)
        return gate_logits


class SimpleCLIP(nn.Module):
    """Simplified CLIP model with fixed text prompts and learnable gating."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        
        self.image_encoder = clip_model.visual
        self.text_encoder_clip = clip_model
        self.simple_text_encoder = SimpleTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.classnum = len(classnames)
        self.classnames = classnames
        
        # Get fixed text features
        templates = cfg.get('templates', ["a photo of a"])
        if isinstance(templates, str):
            templates = [templates]
        self.text_features = self.simple_text_encoder(classnames, templates)
        
        # Gating network configuration
        self.use_rl = bool(cfg.get('use_rl', False))
        self.lambda_sparsity = float(cfg.get('lambda_sparsity', 1e-3))
        self.lambda_minimal = float(cfg.get('lambda_minimal', 1e-2))
        
        # Determine local feature dimension
        try:
            local_dim = self.image_encoder.output_dim
        except Exception:
            local_dim = clip_model.ln_final.weight.shape[0]
        
        gate_hidden = int(cfg.get('gate_hidden', 256))
        self.gating = GatingNetwork(local_dim, gate_hidden)
        
        # Baseline for policy gradient
        self.register_buffer('baseline', torch.tensor(0.0))
        
        print(f"SimpleCLIP initialized with {self.classnum} classes")
        print(f"Text features shape: {self.text_features.shape}")
        print(f"Using templates: {templates}")

    def forward(self, image, labels=None):
        """
        Forward pass.
        
        Args:
            image: [B, 3, H, W] input images
            labels: [B] optional class labels
        
        Returns:
            logits: [B, num_classes] global logits
            logits_local: [B, T, num_classes] local logits
            aux_loss: dict with auxiliary losses (soft mode)
            rl_info: dict with RL info (RL mode)
        """
        device = image.device
        
        # Image encoding
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))
        
        # Normalize
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
        local_image_features = local_image_features / (local_image_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Text features
        text_features = self.text_features.to(device).type(self.dtype)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        logit_scale = self.logit_scale.exp()
        
        # Global logits
        logits = logit_scale * image_features @ text_features.T  # [B, num_classes]
        
        # Gating
        rl_info = {}
        aux_loss = {}
        
        B, T, C = local_image_features.shape
        
        if labels is not None:
            labels = labels.to(device)
            selected_text = text_features[labels]  # [B, C]
        else:
            with torch.no_grad():
                pseudo = logits.argmax(dim=1)
            selected_text = text_features[pseudo]  # [B, C]
        
        # Gating network forward
        gate_logits = self.gating(local_image_features, selected_text)  # [B, T]
        
        if self.use_rl:
            probs = torch.sigmoid(gate_logits)
            m = Bernoulli(probs=probs)
            mask_sample = m.sample()
            log_prob = m.log_prob(mask_sample).sum(dim=1)
            mask_apply = mask_sample.detach()
            mask_ratio = mask_apply.mean(dim=1)
            rl_info['log_prob'] = log_prob
            rl_info['mask'] = mask_sample
            rl_info['mask_ratio'] = mask_ratio
        else:
            probs = torch.sigmoid(gate_logits)
            mask_apply = probs
            mask_ratio = mask_apply.mean()
            aux_loss['sparsity'] = mask_ratio
        
        # Apply gating to local features
        gated_local = local_image_features * mask_apply.unsqueeze(-1)  # [B, T, C]
        
        # Local logits
        logits_local = logit_scale * gated_local @ text_features.T  # [B, T, num_classes]
        
        # Compute negative logits for minimal set constraint
        if labels is not None:
            all_idx = torch.arange(self.classnum, device=device)
            negative_list = []
            for i in range(B):
                neg_idx = torch.cat([all_idx[:labels[i]], all_idx[labels[i]+1:]]) if self.classnum > 1 else torch.tensor([], device=device, dtype=torch.long)
                if neg_idx.numel() == 0:
                    negative_list.append(torch.zeros((T, 0), device=device))
                else:
                    negative_list.append(logits_local[i, :, neg_idx])
            negative_logits = torch.stack([neg for neg in negative_list], dim=0)  # [B, T, num_classes-1]
        else:
            negative_logits = logits_local
        
        if not self.use_rl:
            minimal_loss = torch.mean(negative_logits ** 2)
            aux_loss['minimal'] = minimal_loss
        else:
            with torch.no_grad():
                prob_all = F.softmax(logits_local, dim=-1)
            prob_neg_mean_list = []
            for i in range(B):
                if labels is None:
                    gt = logits[i].argmax()
                else:
                    gt = labels[i]
                mask_idx = torch.ones(self.classnum, device=device, dtype=torch.bool)
                mask_idx[gt] = False
                if mask_idx.sum() == 0:
                    prob_neg_mean_list.append(torch.tensor(0.0, device=device))
                else:
                    prob_neg_mean_list.append(prob_all[i, :, mask_idx].mean())
            confusion_penalty = torch.stack(prob_neg_mean_list).mean()
            rl_info['confusion_penalty'] = confusion_penalty
        
        return logits, logits_local, aux_loss, rl_info
