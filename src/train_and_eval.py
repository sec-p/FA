"""
Unified Training and Evaluation Script with Per-Epoch OOD Testing
Trains modular models with automatic ImageNet validation and OOD evaluation on each epoch.
Optimized for Linux servers with checkpoint savings for learnable parameters only.
Now supports Automatic Mixed Precision (AMP) for stability and speed.
"""

imagenet_templates = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.',
    'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
    'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.',
    'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.',
    'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.',
    'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.',
    'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.',
    'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.',
    'a photo of the dirty {}.', 'a jpeg of a {}.', 'a blurry photo of the {}.',
    'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
    'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.',
    'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
    'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
    'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.',
    'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.',
    'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
    'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.',
    'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.',
    'itap of the {}.', 'a jpeg of the {}.', 'a good photo of a {}.',
    'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.',
    'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.',
    'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
    'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.',
    'graffiti of the {}.', 'a toy {}.', 'itap of my {}.',
    'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
]
# 在 _cache_text_features 中，你的代码已经写好了 mean() 逻辑，
# 只要传入这个长列表，性能马上回升 3-5 个点。


import os
import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast # <--- 关键引入：自动混合精度
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import clip
from src.model_modular import build_modular_model
from my_dataset import build_dataset
from my_dataset.utils import build_data_loader
from src.utils import Logger, cls_acc
import math

class TrainEvalOrchestrator:
    """Orchestrates training and evaluation with per-epoch OOD testing."""
    
    def __init__(self, method: str, epochs: int, lr: float, 
                 batch_size: int, seed: int, device: torch.device,
                 selector_type: str = None, fuser_type: str = None, id_dataset: str = 'imagenet',
                 root_path: str = './data', shots: int = 16, lambda_llm_negatives: float = 0.1,
                 lambda_mixup: float = 0.1, margin: float = 0.2, num_select: int = 16,
                 backbone: str = 'ViT-L/14', class_negatives_path: str = '', use_full_data: bool = False,
                 # 新增参数用于 Slot/STE 调优
                 selector_temperature: float = 1.0,
                 patches_per_slot_attn: int = 16,
                 # OOD score parameters
                 score_type: str = 'GL-MCM',
                 temperature: float = 1.0,
                 lambda_local: float = 1.0):
        
        self.method = method
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        
        # Model components
        self.selector_type = selector_type
        self.fuser_type = fuser_type
        
        # Dataset settings
        self.id_dataset = id_dataset
        self.root_path = root_path
        self.shots = shots
        self.use_full_data = use_full_data
        
        # Loss function coefficients (hyperparameters)
        self.lambda_llm_negatives = lambda_llm_negatives
        self.lambda_mixup = lambda_mixup
        self.margin = margin
        
        # OOD score parameters
        self.score_type = score_type
        self.temperature = temperature
        self.lambda_local = lambda_local
        
        # Model settings
        self.num_select = num_select
        self.backbone = backbone
        self.class_negatives_path = class_negatives_path
        
        # Advanced settings
        self.selector_temperature = selector_temperature
        self.patches_per_slot_attn = patches_per_slot_attn
        
        # Setup random seeds
        self._setup_seed(seed)
        
        # Classnames will be set in setup_data
        self.classnames = []
        
        # Setup logging
        self.log_dir = self._setup_logging()
        self.logger = Logger(os.path.join(self.log_dir, 'training.log'))
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None # For AMP
        self.train_loader = None
        self.test_loader = None
        self.ood_loaders = {}
        self.classnames = []
        
    def _setup_seed(self, seed: int):
        """Setup random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Fix for reproducibility

    def _setup_logging(self) -> str:
        """Setup logging directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'results/{self.method}_{self.selector_type}_{self.seed}_{timestamp}'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f'{log_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{log_dir}/vis', exist_ok=True) # For visualizations
        return log_dir
    
    def setup_data(self):
        """Setup all data loaders (train, ID test, OOD tests)."""
        self.logger.log('Setting up data...')
        
        # Setup transforms
        train_transform = self._get_train_transform()
        test_transform = self._get_test_transform()
        
        # Load ID dataset
        use_full_data = self.use_full_data
        shots = -1 if use_full_data else self.shots
        
        try:
            id_dataset = build_dataset(self.id_dataset, self.root_path, shots)
        except Exception as e:
            self.logger.log(f"Error loading dataset {self.id_dataset}: {e}")
            raise e
        
        # Setup training data loader
        train_data = id_dataset.train_x
        test_data = id_dataset.test if len(id_dataset.test) > 0 else id_dataset.val
        
        # Load class negatives
        self.class_negatives = self._load_class_negatives()
        if self.class_negatives:
            self.logger.log(f"Loaded class negatives for {len(self.class_negatives)} classes")
        else:
            self.logger.log("No class negatives loaded or file not found.")
        
        self.train_loader = build_data_loader(
            data_source=train_data,
            batch_size=self.batch_size,
            tfm=train_transform,
            is_train=True,
            shuffle=True
        )
        
        self.test_loader = build_data_loader(
            data_source=test_data,
            batch_size=self.batch_size,
            is_train=False,
            tfm=test_transform,
            shuffle=False
        )
        
        # Setup OOD data loaders
        OOD_DATASETS = ['iNaturalist', 'SUN', 'Places', 'Textures']
        # You can reduce this list for faster debugging
        # OOD_DATASETS = ['iNaturalist'] 
        
        for ood_dataset in OOD_DATASETS:
            try:
                ood_data = build_dataset(ood_dataset, self.root_path, -1)
                ood_loader = build_data_loader(
                    data_source=ood_data.test if len(ood_data.test) > 0 else ood_data.val,
                    batch_size=self.batch_size,
                    is_train=False,
                    tfm=test_transform,
                    shuffle=False
                )
                self.ood_loaders[ood_dataset] = ood_loader
                self.logger.log(f'  ✓ Loaded OOD dataset: {ood_dataset}')
            except Exception as e:
                self.logger.log(f'  ⚠ Failed to load OOD dataset {ood_dataset}: {e}')
        
        self.classnames = id_dataset.classnames
        self.logger.log(f'  ✓ Training samples: {len(train_data)}')
        self.logger.log(f'  ✓ ID test samples: {len(test_data)}')
    
    def _load_class_negatives(self) -> Dict:
        """Load class negatives from file."""
        class_negatives = {}
        neg_path = self.class_negatives_path
        if neg_path and os.path.exists(neg_path):
            with open(neg_path, 'r') as f:
                class_negatives = json.load(f)
            return class_negatives
        
        # Fallback to default location
        fallback = os.path.join(os.path.dirname(__file__), 'my_dataset', 'class_negatives.json')
        # Also check relative to src
        fallback_src = os.path.join(project_root, 'configs', 'class_negatives.json')
        
        if os.path.exists(fallback):
            with open(fallback, 'r') as f:
                return json.load(f)
        elif os.path.exists(fallback_src):
            with open(fallback_src, 'r') as f:
                return json.load(f)
                
        return {}
    
    def _get_train_transform(self):
        """Get training transforms."""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1),
                                        interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])
    
    def _get_test_transform(self):
        """Get test transforms."""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])
    
    def setup_model(self):
        """Initialize model, optimizer, and scheduler."""
        self.logger.log('Setting up model...')
        
        # Load CLIP
        clip_model, _ = clip.load(self.backbone, device=self.device)
        # Handle DataParallel if needed
        # clip_model = nn.DataParallel(clip_model).to(self.device) 
        # clip_model = clip_model.module
        
        # Create config dict
        cfg = {
            'device': self.device,
            'selector_type': self.selector_type,
            'num_select': self.num_select,
            'fuser_type': self.fuser_type,
            'lambda_llm_negatives': self.lambda_llm_negatives,
            'lambda_mixup': self.lambda_mixup,
            'margin': self.margin,
            'selector_temperature': self.selector_temperature,
            'patches_per_slot_attn': self.patches_per_slot_attn,
            'templates': imagenet_templates,
            
            # Feature flags - Turn on everything for "Full Method"
            'use_redundancy_loss': True,
            'use_llm_negatives': True if self.lambda_llm_negatives > 0 else False,
            'use_semantic_exclusion': True,
            'use_mixup_invariance': True if self.lambda_mixup > 0 else False,
            
            # Loss weights
            'lambda_redundancy': 0.1,
        }
        
        self.logger.log(f'Config: {json.dumps(cfg, default=str, indent=2)}')
        
        # Build modular model
        self.model = build_modular_model(cfg, self.classnames, clip_model, class_negatives=self.class_negatives)
        self.model = self.model.to(self.device)
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = self._get_trainable_params()
        
        if not trainable_params:
            self.logger.log("WARNING: No trainable parameters found! Check your model configuration.")
        
        self.optimizer = torch.optim.SGD(
            trainable_params,
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-5
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # Setup GradScaler for AMP (关键修改)
        self.scaler = GradScaler()
        
        self.logger.log(f'  ✓ Model: {self.method}')
        self.logger.log(f'  ✓ Trainable parameters: {sum(p.numel() for p in trainable_params)}')
    
    def _get_trainable_params(self):
        """Get only trainable parameters."""
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with AMP (Automatic Mixed Precision)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Tqdm bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]', ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # 1. Prepare Data
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            negative_text_tokens = None  # No longer needed since we use cached negative features
            
            self.optimizer.zero_grad()
            
            # 2. Forward with Autocast (Mixed Precision) - 关键修改
            # 模型的大部分计算(Conv, Linear)在FP16下进行，Loss计算会自动(通过装饰器)切回FP32
            with autocast():
                output_dict = self.model(images, labels=labels, negative_text_tokens=negative_text_tokens)
                
                logits = output_dict['logits']
                aux_losses = output_dict['aux_losses']
                
                # Main Loss
                ce_loss = F.cross_entropy(logits, labels)
                
                # Aux Losses
                total_aux_loss = 0.0
                for k, v in aux_losses.items():
                    if v.requires_grad:
                        total_aux_loss += v
                
                loss = ce_loss + total_aux_loss

            # 3. Backward with Scaler - 关键修改
            # Scaler 会自动处理梯度下溢(underflow)问题，无需手动检查 NaN
            self.scaler.scale(loss).backward()

            # 4. Gradient Clipping (must unscale first)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 5. Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 6. Metrics & Logging
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            # Prepare loss info for display
            loss_info = {
                'Loss': f"{loss.item():.4f}",
                'CE': f"{ce_loss.item():.4f}",
                'Acc': f"{100.*correct/total:.2f}%"
            }
            
            # Add auxiliary losses to display
            for k, v in aux_losses.items():
                if v.requires_grad:
                    loss_info[k] = f"{v.item():.4f}"
            
            # Update progress bar with detailed loss info
            pbar.set_postfix(loss_info)
            
            # Log detailed loss info occasionally
            if batch_idx % 50 == 0:
                loss_str = f"CE: {ce_loss.item():.4f}"
                for k, v in aux_losses.items():
                    loss_str += f", {k}: {v.item():.4f}"
                self.logger.log(f"  Batch {batch_idx}: {loss_str}")

        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        train_acc = 100.0 * correct / total if total > 0 else 0
        
        return avg_loss, train_acc, True
    
    def evaluate_id(self) -> float:
        """Evaluate on ID (ImageNet) test set."""

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Inference only needs images (labels are for metric calc only)
                with autocast(): # 推理也使用FP16加速
                    output_dict = self.model(images, labels=None) # No labels passed = Inference mode
                    logits = output_dict['logits']
                
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        id_acc = 100.0 * correct / total if total > 0 else 0.0
        return id_acc
    
    def evaluate_ood_dataset(self, dataset_name: str) -> Tuple[float, float]:
        """Evaluate on single OOD dataset."""
        if dataset_name not in self.ood_loaders:
            return 0.0, 0.0
        
        self.model.eval()
        
        # 1. Get ID confidences
        to_np = lambda x: x.data.cpu().numpy()
        id_scores = []
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    images = batch['images'].to(self.device)
                else:
                    images, _ = batch
                    images = images.to(self.device)
                
                with autocast():
                    output_dict = self.model(images)
                                        
                    # Extract features similar to get_ood_scores_clip
                    global_features = output_dict['global_features']
                    local_features = output_dict['local_features']
                    text_feats = self.model._text_features.to(self.device)
                    
                    # Normalize features
                    # global_features = F.normalize(global_features, dim=-1)
                    global_features = global_features / global_features.norm(dim=-1, keepdim=True)
                    # local_features = F.normalize(local_features, dim=-1)
                    local_features = local_features / local_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate logits (no logit_scale, consistent with GL-MCM)
                    output_global = global_features @ text_feats.T
                    output_local = local_features @ text_feats.T
                    
                    # Calculate scores based on score_type
                    if self.score_type == 'energy':
                        # Energy = - T * logsumexp(logit_k / T)
                        scores = -self.temperature * torch.logsumexp(output_global / self.temperature, dim=1)
                    elif self.score_type == 'entropy':
                        smax_global = F.softmax(output_global / self.temperature, dim=1)
                        # Convert to numpy for entropy calculation
                        smax_global_np = smax_global.cpu().numpy()
                        scores_np = entropy(smax_global_np, axis=1)
                        scores = torch.tensor(scores_np, device=self.device)
                    elif self.score_type == 'var':
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        # Convert to numpy for variance calculation
                        smax_global_np = smax_global.cpu().numpy()
                        scores_np = -np.var(smax_global_np, axis=1)
                        scores = torch.tensor(scores_np, device=self.device)
                    elif self.score_type == 'MCM':
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        scores = -np.max(smax_global, dim=1)[0]
                    elif self.score_type == 'max-logit':
                        scores = -np.max(output_global, axis=1)[0]
                    elif self.score_type == 'L-MCM':
                        # Local MCM - calculate softmax over local features
                        smax_local = to_np(F.softmax(output_local / self.temperature, dim=1))
                        # For ViT, local_features shape is (B, N, D) where N is number of patches
                        # We need to reshape to match GL-MCM's expected shape (B, H, W, C) for spatial dimensions
                        B, N, C = smax_local.shape
                        # Assume square patches
                        H = W = int(math.sqrt(N))
                        smax_local_reshaped = smax_local.reshape(B, H, W, C)
                        # Get max over spatial dimensions (H, W)
                        scores = -np.max(smax_local_reshaped, axis=(1, 2, 3))[0]
                    elif self.score_type == 'GL-MCM':
                        import pdb
                        pdb.set_trace()
                        # Global-Local MCM combination
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        mcm_global_score = -np.max(smax_global, axis=1)[0]
                        
                        # Local MCM component
                        smax_local = to_np(F.softmax(output_local / self.temperature, dim=1))
                        B, N, C = smax_local.shape
                        H = W = int(math.sqrt(N))
                        smax_local_reshaped = smax_local.reshape(B, H, W, C)
                        mcm_local_score = -np.max(smax_local_reshaped, axis=(1, 2, 3))[0]
                        
                        # Combine with lambda_local weight
                        scores = mcm_global_score + self.lambda_local * mcm_local_score
                
                
                id_scores.extend(scores.cpu().numpy())
        
        # 2. Get OOD confidences
        ood_scores = []
        with torch.no_grad():
            for batch in self.ood_loaders[dataset_name]:
                if isinstance(batch, dict):
                    images = batch['images'].to(self.device)
                else:
                    images, _ = batch
                    images = images.to(self.device)
                
                with autocast():
                    output_dict = self.model(images)
                                        
                    # Extract features similar to get_ood_scores_clip
                    global_features = output_dict['global_features']
                    local_features = output_dict['local_features']
                    text_feats = self.model._text_features.to(self.device)
                    
                    # Normalize features
                    global_features = F.normalize(global_features, dim=-1)
                    local_features = F.normalize(local_features, dim=-1)
                    
                    # Calculate logits (no logit_scale, consistent with GL-MCM)
                    output_global = global_features @ text_feats.T
                    output_local = local_features @ text_feats.T
                    
                    # Calculate scores based on score_type
                    if self.score_type == 'energy':
                        # Energy = - T * logsumexp(logit_k / T)
                        scores = -self.temperature * torch.logsumexp(output_global / self.temperature, dim=1)
                    elif self.score_type == 'entropy':
                        smax_global = F.softmax(output_global / self.temperature, dim=1)
                        # Convert to numpy for entropy calculation
                        smax_global_np = smax_global.cpu().numpy()
                        scores_np = entropy(smax_global_np, axis=1)
                        scores = torch.tensor(scores_np, device=self.device)
                    elif self.score_type == 'var':
                        smax_global = F.softmax(output_global / self.temperature, dim=1)
                        # Convert to numpy for variance calculation
                        smax_global_np = smax_global.cpu().numpy()
                        scores_np = -np.var(smax_global_np, axis=1)
                        scores = torch.tensor(scores_np, device=self.device)
                    elif self.score_type in ['MCM', 'max-logit']:
                        if self.score_type == 'max-logit':
                            # For max-logit, don't apply softmax
                            smax_global = output_global
                        else:
                            # For MCM, apply softmax
                            smax_global = F.softmax(output_global / self.temperature, dim=1)
                        scores = -torch.max(smax_global, dim=1)[0]
                    elif self.score_type == 'L-MCM':
                        # Local MCM - calculate softmax over local features
                        smax_local = F.softmax(output_local / self.temperature, dim=1)
                        # For ViT, local_features shape is (B, N, D) where N is number of patches
                        # We need to reshape to match GL-MCM's expected shape (B, H, W, C) for spatial dimensions
                        B, N, C = smax_local.shape
                        # Assume square patches
                        H = W = int(math.sqrt(N))
                        smax_local_reshaped = smax_local.reshape(B, H, W, C)
                        # Get max over spatial dimensions (H, W)
                        scores = -torch.max(smax_local_reshaped, dim=(1, 2, 3))[0]
                    elif self.score_type == 'GL-MCM':
                        # Global-Local MCM combination
                        smax_global = F.softmax(output_global / self.temperature, dim=1)
                        mcm_global_score = -torch.max(smax_global, dim=1)[0]
                        
                        # Local MCM component
                        smax_local = F.softmax(output_local / self.temperature, dim=1)
                        B, N, C = smax_local.shape
                        H = W = int(math.sqrt(N))
                        smax_local_reshaped = smax_local.reshape(B, H, W, C)
                        mcm_local_score = -torch.max(smax_local_reshaped, dim=(1, 2, 3))[0]
                        
                        # Combine with lambda_local weight
                        scores = mcm_global_score + self.lambda_local * mcm_local_score
                
                
                ood_scores.extend(scores.cpu().numpy())
        
        # 3. Compute metrics
        if len(id_scores) == 0 or len(ood_scores) == 0:
            return 0.0, 0.0
            
        id_scores = np.array(id_scores)
        ood_scores = np.array(ood_scores)
        
        auroc = self._compute_auroc(id_scores, ood_scores)
        fpr95 = self._compute_fpr95(id_scores, ood_scores)
        
        return auroc, fpr95
    
    def _compute_auroc(self, id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])
        try:
            auroc = roc_auc_score(labels, scores) * 100
        except:
            auroc = 0.0
        return auroc
    
    def _compute_fpr95(self, id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        id_sorted = np.sort(id_scores)
        # We want to keep 95% of ID samples, so threshold is at the 5th percentile
        thresh_idx = int(len(id_scores) * 0.05)
        threshold = id_sorted[thresh_idx]
        
        # FPR = FP / N = sum(ood >= thresh) / len(ood)
        fpr = np.sum(ood_scores >= threshold) / len(ood_scores)
        return fpr * 100
    
    def evaluate_epoch(self, epoch: int) -> Dict:
        """Evaluate on ID and all OOD datasets."""
        results = {}
        
        # ID evaluation
        id_acc = self.evaluate_id()
        results['id_accuracy'] = id_acc
        
        # OOD evaluations
        ood_aurocs = []
        ood_fpr95s = []
        
        for ood_name in self.ood_loaders.keys():
            auroc, fpr95 = self.evaluate_ood_dataset(ood_name)
            results[f'{ood_name}_auroc'] = auroc
            results[f'{ood_name}_fpr95'] = fpr95
            ood_aurocs.append(auroc)
            ood_fpr95s.append(fpr95)
        
        # Averages
        results['avg_ood_auroc'] = np.mean(ood_aurocs) if ood_aurocs else 0.0
        results['avg_ood_fpr95'] = np.mean(ood_fpr95s) if ood_fpr95s else 0.0
        
        return results
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save checkpoint with only trainable parameters."""
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', f'epoch_{epoch:03d}.pt')
        
        # Extract only trainable parameters
        trainable_state = {}
        for name, param in self.model.named_parameters():
            # Save if requires grad OR if it's a buffer like running_mean
            if param.requires_grad:
                trainable_state[name] = param.data.clone()
        
        checkpoint = {
            'epoch': epoch,
            'method': self.method,
            'seed': self.seed,
            'state_dict': trainable_state,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(), # Save scaler state
            'metrics': metrics,
            'config': {
                'selector_type': self.selector_type,
                'fuser_type': self.fuser_type,
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def save_visualization(self, epoch):
        pass

    def train_with_eval(self):
        """Main training loop."""
        self.logger.log('\n' + '='*80)
        self.logger.log(f'Starting training: {self.method} (seed={self.seed})')
        self.logger.log('='*80 + '\n')
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        best_avg_auroc = 0.0
        results_history = []
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc, _ = self.train_epoch(epoch)
            self.scheduler.step()
            

            eval_results={}
            
            # Log results
            self.logger.log(f'\n[Epoch {epoch+1}/{self.epochs}]')
            self.logger.log(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

            if (epoch+1)%5==0 and epoch+1>0:
                # Evaluate
                eval_results = self.evaluate_epoch(epoch)
                self.logger.log(f'  ID Accuracy: {eval_results["id_accuracy"]:.2f}%')
                
                for ood_name in self.ood_loaders.keys():
                    auroc = eval_results[f'{ood_name}_auroc']
                    fpr95 = eval_results[f'{ood_name}_fpr95']
                    self.logger.log(f'  {ood_name:15} AUROC: {auroc:.2f}%, FPR95: {fpr95:.2f}%')
                
                avg_auroc = eval_results["avg_ood_auroc"]
                avg_fpr95 = eval_results["avg_ood_fpr95"]
                self.logger.log(f'  Avg OOD AUROC: {avg_auroc:.2f}%, Avg OOD FPR95: {avg_fpr95:.2f}%')
            
                # Save checkpoint (Every 5 epochs or best)
                if (epoch % 5 == 0 or epoch == self.epochs - 1) and epoch+1>10:
                    self.save_checkpoint(epoch, eval_results)
            
                # Track best
                if avg_auroc > best_avg_auroc:
                    best_avg_auroc = avg_auroc
                    self.save_checkpoint(999, eval_results) # 999 as code for 'best'
                    self.logger.log(f"  ★ New Best Avg AUROC: {best_avg_auroc:.2f}%")
            
            # Update history
            results_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                **eval_results
            })
            
            # Dump history
            with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
                json.dump(results_history, f, indent=2)
                
        self.logger.log('\nTraining completed.')


def main():
    parser = argparse.ArgumentParser(description='Train modular OOD detection with AMP')
    
    # Basic Config
    parser.add_argument('--method', type=str, default='custom')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005) 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    # Components
    parser.add_argument('--selector_type', type=str, default='slot', help="'mlp' or 'slot'")
    parser.add_argument('--fuser_type', type=str, default='query_attn')
    
    # Dataset
    parser.add_argument('--id_dataset', type=str, default='imagenet')
    parser.add_argument('--root_path', type=str, default='/data/datasets')
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--use_full_data', action='store_true')
    
    # Loss Weights
    parser.add_argument('--lambda_llm_negatives', type=float, default=0.1)
    parser.add_argument('--lambda_mixup', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.2)
    
    # Paths
    parser.add_argument('--class_negatives_path', type=str, default='configs/class_negatives_clean.json')
    parser.add_argument('--backbone', type=str, default='ViT-B/16')

    # === 关键修正：确保包含 num_select 参数 ===
    parser.add_argument('--num_select', type=int, default=16,
                        help='Number of tokens/features to retain in selector (k)')
    # ========================================

    # Advanced params
    parser.add_argument('--selector_temperature', type=float, default=1.0)
    parser.add_argument('--patches_per_slot_attn', type=int, default=16)

    args = parser.parse_args()

    # Create trainer
    trainer = TrainEvalOrchestrator(
        method=args.method,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=torch.device(args.device),
        selector_type=args.selector_type,
        fuser_type=args.fuser_type,
        id_dataset=args.id_dataset,
        root_path=args.root_path,
        shots=args.shots,
        lambda_llm_negatives=args.lambda_llm_negatives,
        lambda_mixup=args.lambda_mixup,
        margin=args.margin,
        backbone=args.backbone,
        class_negatives_path=args.class_negatives_path,
        use_full_data=args.use_full_data,
        num_select=args.num_select,
        selector_temperature=args.selector_temperature,
        patches_per_slot_attn=args.patches_per_slot_attn
    )

    trainer.train_with_eval()


if __name__ == '__main__':
    main()