"""
Training script for Modular OOD Detection Framework.
Supports flexible loss combinations and ablation studies.
"""

import os
import sys
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import clip
from model_modular import build_modular_model
from my_dataset import build_dataset
from my_dataset.utils import build_data_loader
from utils import Logger, cls_acc
import json


class ModularTrainer:
    """Trainer for modular OOD detection framework."""
    
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.epoch = 0
        
    def setup_data(self):
        """Setup data loaders."""
        # Transforms
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()
        
        # Dataset
        use_full_data = self.cfg.get('use_full_data', False)
        shots = -1 if use_full_data else self.cfg.get('shots', 16)
        
        few_shot_dataset = build_dataset(
            self.cfg['id_dataset'], 
            self.cfg['root_path'], 
            shots
        )
        
        batch_size = self.cfg['fine_tune_batch_size']
        
        if use_full_data:
            # Combine train + val for full dataset
            train_data = few_shot_dataset.train_x + few_shot_dataset.val
        else:
            train_data = few_shot_dataset.train_x
        
        self.train_loader = build_data_loader(
            data_source=train_data,
            batch_size=batch_size,
            tfm=self.train_transform,
            is_train=True,
            shuffle=True
        )
        
        self.test_loader = build_data_loader(
            data_source=few_shot_dataset.test,
            batch_size=batch_size,
            is_train=False,
            tfm=self.test_transform,
            shuffle=False
        )
        
        self.classnames = few_shot_dataset.classnames
        self.cfg['classnames'] = self.classnames
        
        return few_shot_dataset
    
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
        """Initialize model."""
        clip_model, _ = clip.load(self.cfg['backbone'], device=self.device)
        clip_model = nn.DataParallel(clip_model).to(self.device)
        clip_model = clip_model.module
        
        self.model = build_modular_model(self.cfg, self.classnames, clip_model)
        self.model = nn.DataParallel(self.model).to(self.device)
        
        # Only fuser parameters are trainable
        for name, param in self.model.named_parameters():
            if 'selector' in name or 'fuser' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✓ Model initialized. Trainable params: {trainable_params:,}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.cfg.get('lr_gate', self.cfg['lr'] * 0.1),
            weight_decay=self.cfg.get('weight_decay', 1e-4)
        )
        
        total_steps = self.cfg['fine_tune_train_epoch'] * len(self.train_loader)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"✓ Optimizer initialized. Total steps: {total_steps}")
    
    def compute_losses(self, output: dict, labels: torch.Tensor) -> dict:
        """Compute all losses."""
        losses = {}
        
        # Main classification loss
        ce_loss = self.criterion(output['logits'], labels)
        losses['ce'] = ce_loss
        
        # Auxiliary losses from selector
        if 'diversity' in output['aux_losses']:
            lambda_div = self.cfg.get('lambda_diversity', 0.01)
            losses['diversity'] = lambda_div * output['aux_losses']['diversity']
        
        if 'orthogonality' in output['aux_losses']:
            lambda_ortho = self.cfg.get('lambda_orthogonality', 0.01)
            losses['orthogonality'] = lambda_ortho * output['aux_losses']['orthogonality']
        
        if 'semantic_exclusion' in output['aux_losses']:
            lambda_sem = self.cfg.get('lambda_semantic_exclusion', 0.01)
            losses['semantic_exclusion'] = lambda_sem * output['aux_losses']['semantic_exclusion']
        
        return losses
    
    def train_epoch(self, epoch: int):
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        loss_breakdown = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            output = self.model(images, targets)
            
            # Compute losses
            losses = self.compute_losses(output, targets)
            
            # Total loss
            total_loss_val = sum(losses.values())
            
            # Backward
            self.optimizer.zero_grad()
            total_loss_val.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            acc = cls_acc(output['logits'], targets, topk=1)
            correct += acc / 100 * len(targets)
            total += len(targets)
            total_loss += total_loss_val.item()
            
            # Track loss breakdown
            for k, v in losses.items():
                if k not in loss_breakdown:
                    loss_breakdown[k] = 0.0
                loss_breakdown[k] += v.item()
            
            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{correct / total:.4f}'
            })
        
        train_acc = correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        # Print loss breakdown
        print(f"\nLoss Breakdown:")
        for k, v in loss_breakdown.items():
            print(f"  {k}: {v / len(self.train_loader):.4f}")
        
        return {
            'loss': avg_loss,
            'acc': train_acc,
            'breakdown': loss_breakdown
        }
    
    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Evaluation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                output = self.model(images)
                
                acc = cls_acc(output['logits'], targets, topk=1)
                correct += acc / 100 * len(targets)
                total += len(targets)
        
        test_acc = correct / total
        return test_acc
    
    def train(self, num_epochs: int):
        """Main training loop."""
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            test_acc = self.evaluate()
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['acc']:.4f}, "
                  f"Test Acc: {test_acc:.4f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                self.save_checkpoint()
        
        print(f"\n✓ Training complete. Best test accuracy: {best_acc:.4f}")
        return best_acc
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        os.makedirs(self.cfg['cache_dir'], exist_ok=True)
        checkpoint_path = os.path.join(self.cfg['cache_dir'], 'model_best.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/my_config.yaml')
    parser.add_argument('--is_train', type=int, default=1)
    args = parser.parse_args()
    
    # Load config
    assert os.path.exists(args.config)
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['device'] = device
    
    # Setup cache directory
    import socket
    timestamp = socket.gethostname()
    cache_dir = os.path.join(
        './my_caches',
        cfg['id_dataset'],
        cfg['backbone'].replace('/', '-'),
        f"selector_{cfg.get('selector_type', 'mlp')}_fuser_{cfg.get('fuser_type', 'mean')}",
        # include ood map identifier if provided
        (os.path.splitext(os.path.basename(cfg.get('ood_classmap_path', '')))[0] if cfg.get('ood_classmap_path') else ''),
        f"seed{cfg['seed']}"
    )
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    
    # Logger
    sys.stdout = Logger(os.path.join(cache_dir, 'log_train.txt'), stream=sys.stdout)
    
    # Random seeds
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    print(f"\n{'='*60}")
    print(f"Modular OOD Detection Training")
    print(f"{'='*60}")
    print(f"Config: {cfg}\n")
    
    if args.is_train == 1:
        # Training
        trainer = ModularTrainer(cfg, device)
        
        # Setup
        trainer.setup_data()
        trainer.setup_model()
        # If an OOD mapping JSON is provided, load and inject into model
        ood_map_path = cfg.get('ood_classmap_path', None)
        if ood_map_path and os.path.exists(ood_map_path):
            try:
                with open(ood_map_path, 'r') as f:
                    ood_map = json.load(f)
                # pass mapping to model (handle DataParallel wrapper)
                try:
                    model_wrapper = trainer.model
                    if hasattr(model_wrapper, 'module'):
                        model_wrapper.module.set_ood_classmap(ood_map)
                    else:
                        model_wrapper.set_ood_classmap(ood_map)
                    print(f"✓ Loaded OOD classmap from {ood_map_path}")
                except Exception as e:
                    print(f"Warning: could not attach OOD map to model: {e}")
            except Exception as e:
                print(f"Warning: failed to read OOD JSON {ood_map_path}: {e}")
        trainer.setup_optimizer()
        
        # Train
        num_epochs = cfg.get('fine_tune_train_epoch', 50)
        best_acc = trainer.train(num_epochs)
        
    else:
        print("Evaluation mode not implemented in this script.")


if __name__ == '__main__':
    main()
