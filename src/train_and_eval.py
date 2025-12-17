"""
Unified Training and Evaluation Script with Per-Epoch OOD Testing
Trains modular models with automatic ImageNet validation and OOD evaluation on each epoch.
Optimized for Linux servers with checkpoint savings for learnable parameters only.
"""

import os
import sys
import json
import yaml
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import clip
from model_modular import build_modular_model
from my_dataset import build_dataset
from my_dataset.utils import build_data_loader
from utils import Logger, cls_acc


class TrainingConfig:
    """Configuration for training and evaluation."""
    
    # Method configurations (selector_type:fuser_type)
    METHODS = {
        'baseline_mean': {'selector_type': None, 'fuser_type': 'mean'},
        'baseline_attention': {'selector_type': None, 'fuser_type': 'query_attn'},
        'selector_mlp': {'selector_type': 'mlp', 'fuser_type': 'mean'},
        'selector_slot': {'selector_type': 'slot', 'fuser_type': 'mean'},
        'fuser_attention': {'selector_type': 'mlp', 'fuser_type': 'query_attn'},
        'full_model': {'selector_type': 'slot', 'fuser_type': 'self_attn'},
    }
    
    # OOD datasets
    OOD_DATASETS = ['sun397', 'dtd', 'eurosat', 'oxford_pets']
    
    # Default hyperparameters
    DEFAULT_EPOCHS = 50
    DEFAULT_LR = 0.001
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_SEED = 42


class TrainEvalOrchestrator:
    """Orchestrates training and evaluation with per-epoch OOD testing."""
    
    def __init__(self, config_path: str, method: str, epochs: int, lr: float, 
                 batch_size: int, seed: int, device: torch.device):
        self.config_path = config_path
        self.method = method
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        
        # Setup random seeds
        self._setup_seed(seed)
        
        # Load base config
        self.cfg = self._load_config(config_path)
        
        # Update config with method and hyperparameters
        self._apply_method_config(method)
        self.cfg['fine_tune_batch_size'] = batch_size
        self.cfg['learning_rate'] = lr
        
        # Setup logging
        self.log_dir = self._setup_logging()
        self.logger = Logger(os.path.join(self.log_dir, 'training.log'))
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
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
        
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    
    def _apply_method_config(self, method: str):
        """Apply method-specific configuration."""
        if method not in TrainingConfig.METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(TrainingConfig.METHODS.keys())}")
        
        method_cfg = TrainingConfig.METHODS[method]
        self.cfg['selector_type'] = method_cfg['selector_type']
        self.cfg['fuser_type'] = method_cfg['fuser_type']
    
    def _setup_logging(self) -> str:
        """Setup logging directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'results/{self.method}_{self.seed}_{timestamp}'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f'{log_dir}/checkpoints', exist_ok=True)
        return log_dir
    
    def setup_data(self):
        """Setup all data loaders (train, ID test, OOD tests)."""
        self.logger.log('Setting up data...')
        
        # Setup transforms
        train_transform = self._get_train_transform()
        test_transform = self._get_test_transform()
        
        # Load ID dataset
        use_full_data = self.cfg.get('use_full_data', False)
        shots = -1 if use_full_data else self.cfg.get('shots', 16)
        
        id_dataset = build_dataset(
            self.cfg['id_dataset'],
            self.cfg['root_path'],
            shots
        )
        
        # Setup training data loader
        train_data = id_dataset.train_x if not use_full_data else id_dataset.train_x + id_dataset.val
        test_data = id_dataset.test if len(id_dataset.test) > 0 else id_dataset.val
        
        # Load class negatives
        class_negatives = self._load_class_negatives()
        
        self.train_loader = build_data_loader(
            data_source=train_data,
            batch_size=self.batch_size,
            tfm=train_transform,
            is_train=True,
            shuffle=True,
            class_negatives=class_negatives,
            text_encoder=None,
            device=self.device
        )
        
        self.test_loader = build_data_loader(
            data_source=test_data,
            batch_size=self.batch_size,
            is_train=False,
            tfm=test_transform,
            shuffle=False,
            class_negatives=class_negatives,
            text_encoder=None,
            device=self.device
        )
        
        # Setup OOD data loaders
        for ood_dataset in TrainingConfig.OOD_DATASETS:
            try:
                ood_data = build_dataset(ood_dataset, self.cfg['root_path'], -1)
                ood_loader = build_data_loader(
                    data_source=ood_data.test if len(ood_data.test) > 0 else ood_data.val,
                    batch_size=self.batch_size,
                    is_train=False,
                    tfm=test_transform,
                    shuffle=False,
                    class_negatives={},
                    text_encoder=None,
                    device=self.device
                )
                self.ood_loaders[ood_dataset] = ood_loader
                self.logger.log(f'  ✓ Loaded OOD dataset: {ood_dataset}')
            except Exception as e:
                self.logger.log(f'  ⚠ Failed to load OOD dataset {ood_dataset}: {e}')
        
        self.classnames = id_dataset.classnames
        self.cfg['classnames'] = self.classnames
        self.logger.log(f'  ✓ Training samples: {len(train_data)}')
        self.logger.log(f'  ✓ ID test samples: {len(test_data)}')
        self.logger.log(f'  ✓ OOD loaders: {len(self.ood_loaders)}')
    
    def _load_class_negatives(self) -> Dict:
        """Load class negatives from file."""
        class_negatives = {}
        
        # Try config path first
        neg_path = self.cfg.get('class_negatives_path', '')
        if neg_path and os.path.exists(neg_path):
            with open(neg_path, 'r') as f:
                class_negatives = json.load(f)
            return class_negatives
        
        # Fallback to default location
        fallback = os.path.join(os.path.dirname(__file__), 'my_dataset', 'class_negatives.json')
        if os.path.exists(fallback):
            try:
                with open(fallback, 'r') as f:
                    class_negatives = json.load(f)
            except Exception:
                pass
        
        return class_negatives
    
    def _get_train_transform(self):
        """Get training transforms."""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1),
                                        interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
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
        clip_model, _ = clip.load(self.cfg['backbone'], device=self.device)
        clip_model = nn.DataParallel(clip_model).to(self.device)
        clip_model = clip_model.module
        
        # Build modular model
        self.model = build_modular_model(self.cfg, self.classnames, clip_model)
        self.model = self.model.to(self.device)
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = self._get_trainable_params()
        self.optimizer = torch.optim.SGD(
            trainable_params,
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-5
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        self.logger.log(f'  ✓ Model: {self.method}')
        self.logger.log(f'  ✓ Optimizer: SGD (lr={self.lr})')
        self.logger.log(f'  ✓ Scheduler: CosineAnnealingLR')
        self.logger.log(f'  ✓ Trainable parameters: {sum(p.numel() for p in trainable_params)}')
    
    def _get_trainable_params(self):
        """Get only trainable parameters (skip CLIP backbone)."""
        trainable_params = []
        for name, param in self.model.named_parameters():
            # Skip CLIP parameters
            if 'visual' not in name and 'transformer' not in name:
                if param.requires_grad:
                    trainable_params.append(param)
        return trainable_params
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]', ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # Extract batch data based on loader type
            if isinstance(batch, dict):
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
            else:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            
            # Compute loss
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.update(1)
        
        avg_loss = total_loss / len(self.train_loader)
        train_acc = 100.0 * correct / total
        
        return avg_loss, train_acc
    
    def evaluate_id(self) -> float:
        """Evaluate on ID (ImageNet) test set."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    images = batch['images'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                
                logits = self.model(images)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        id_acc = 100.0 * correct / total
        return id_acc
    
    def evaluate_ood_dataset(self, dataset_name: str) -> Tuple[float, float]:
        """
        Evaluate on single OOD dataset.
        Returns: AUROC (%), FPR95 (%)
        """
        if dataset_name not in self.ood_loaders:
            return 0.0, 0.0
        
        self.model.eval()
        
        # Get ID confidences
        id_scores = []
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    images = batch['images'].to(self.device)
                else:
                    images, _ = batch
                    images = images.to(self.device)
                
                logits = self.model(images)
                scores = F.softmax(logits, dim=1).max(1)[0]
                id_scores.extend(scores.cpu().numpy())
        
        # Get OOD confidences
        ood_scores = []
        with torch.no_grad():
            for batch in self.ood_loaders[dataset_name]:
                if isinstance(batch, dict):
                    images = batch['images'].to(self.device)
                else:
                    images, _ = batch
                    images = images.to(self.device)
                
                logits = self.model(images)
                scores = F.softmax(logits, dim=1).max(1)[0]
                ood_scores.extend(scores.cpu().numpy())
        
        # Compute metrics
        id_scores = np.array(id_scores)
        ood_scores = np.array(ood_scores)
        
        auroc = self._compute_auroc(id_scores, ood_scores)
        fpr95 = self._compute_fpr95(id_scores, ood_scores)
        
        return auroc, fpr95
    
    def _compute_auroc(self, id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        """Compute AUROC."""
        from sklearn.metrics import roc_auc_score
        
        labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])
        
        auroc = roc_auc_score(labels, scores) * 100
        return auroc
    
    def _compute_fpr95(self, id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        """Compute FPR at 95% TPR."""
        # Sort scores
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        
        # Compute TPR at different thresholds
        thresholds = np.sort(scores)[::-1]
        
        for threshold in thresholds:
            tpr = np.sum((id_scores >= threshold)) / len(id_scores)
            if tpr <= 0.95:
                fpr = np.sum((ood_scores >= threshold)) / len(ood_scores)
                return fpr * 100
        
        return 0.0
    
    def evaluate_epoch(self, epoch: int) -> Dict:
        """Evaluate on ID and all OOD datasets."""
        results = {}
        
        # ID evaluation
        id_acc = self.evaluate_id()
        results['id_accuracy'] = id_acc
        
        # OOD evaluations
        ood_aurocs = []
        ood_fpr95s = []
        
        for ood_dataset in TrainingConfig.OOD_DATASETS:
            auroc, fpr95 = self.evaluate_ood_dataset(ood_dataset)
            results[f'{ood_dataset}_auroc'] = auroc
            results[f'{ood_dataset}_fpr95'] = fpr95
            ood_aurocs.append(auroc)
            ood_fpr95s.append(fpr95)
        
        # Averages
        results['avg_ood_auroc'] = np.mean(ood_aurocs)
        results['avg_ood_fpr95'] = np.mean(ood_fpr95s)
        
        return results
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save checkpoint with only trainable parameters."""
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', f'epoch_{epoch:03d}.pt')
        
        # Extract only trainable parameters
        trainable_state = {}
        for name, param in self.model.named_parameters():
            if 'visual' not in name and 'transformer' not in name:
                if param.requires_grad:
                    trainable_state[name] = param.data.clone()
        
        checkpoint = {
            'epoch': epoch,
            'method': self.method,
            'seed': self.seed,
            'trainable_state_dict': trainable_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def train_with_eval(self):
        """Main training loop with per-epoch evaluation."""
        self.logger.log('\n' + '='*80)
        self.logger.log(f'Starting training: {self.method} (seed={self.seed})')
        self.logger.log('='*80 + '\n')
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Training loop
        best_id_acc = 0.0
        results_history = []
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate
            eval_results = self.evaluate_epoch(epoch)
            
            # Log results
            self.logger.log(f'\n[Epoch {epoch+1}/{self.epochs}]')
            self.logger.log(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            self.logger.log(f'  ID Accuracy: {eval_results["id_accuracy"]:.2f}%')
            
            for ood_dataset in TrainingConfig.OOD_DATASETS:
                auroc = eval_results[f'{ood_dataset}_auroc']
                fpr95 = eval_results[f'{ood_dataset}_fpr95']
                self.logger.log(f'  {ood_dataset:15} AUROC: {auroc:.2f}%, FPR95: {fpr95:.2f}%')
            
            self.logger.log(f'  Avg AUROC: {eval_results["avg_ood_auroc"]:.2f}%')
            
            # Save checkpoint
            ckpt_path = self.save_checkpoint(epoch, eval_results)
            self.logger.log(f'  Checkpoint saved: {os.path.basename(ckpt_path)}')
            
            # Update scheduler
            self.scheduler.step()
            
            # Track best
            if eval_results["id_accuracy"] > best_id_acc:
                best_id_acc = eval_results["id_accuracy"]
            
            results_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                **eval_results
            })
        
        # Save final results
        results_file = os.path.join(self.log_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results_history, f, indent=2)
        
        self.logger.log('\n' + '='*80)
        self.logger.log(f'Training completed! Best ID Accuracy: {best_id_acc:.2f}%')
        self.logger.log(f'Results saved to: {results_file}')
        self.logger.log('='*80 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate modular OOD detection')
    parser.add_argument('--config', type=str, default='configs/my_config.yaml',
                       help='Path to config file')
    parser.add_argument('--method', type=str, default='baseline_mean',
                       choices=list(TrainingConfig.METHODS.keys()),
                       help='Training method')
    parser.add_argument('--epochs', type=int, default=TrainingConfig.DEFAULT_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=TrainingConfig.DEFAULT_LR,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=TrainingConfig.DEFAULT_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=TrainingConfig.DEFAULT_SEED,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create trainer
    trainer = TrainEvalOrchestrator(
        config_path=args.config,
        method=args.method,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device
    )
    
    # Train and evaluate
    trainer.train_with_eval()


if __name__ == '__main__':
    main()
