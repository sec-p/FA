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
        self.num_classes = 0
        self.classnames = []
        
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
        
        # Load class negatives if provided. If user did not set a path in config,
        # fall back to a `my_dataset/class_negatives.json` placed by the user.
        class_negatives = {}
        neg_path = self.cfg.get('class_negatives_path', '')
        if neg_path and os.path.exists(neg_path):
            with open(neg_path, 'r') as f:
                class_negatives = json.load(f)
            print(f"✓ Loaded class negatives from {neg_path}")
        else:
            # Fallback to my_dataset/class_negatives.json if exists
            fallback = os.path.join(os.path.dirname(__file__), 'my_dataset', 'class_negatives.json')
            if os.path.exists(fallback):
                try:
                    with open(fallback, 'r') as f:
                        class_negatives = json.load(f)
                    print(f"✓ Loaded class negatives from {fallback}")
                except Exception as e:
                    print(f"Warning: failed to load fallback class_negatives.json: {e}")
        
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
            shuffle=True,
            class_negatives=class_negatives,
            text_encoder=None,  # Will be set in setup_model
            device=self.device
        )
        
        self.test_loader = build_data_loader(
            data_source=few_shot_dataset.test,
            batch_size=batch_size,
            is_train=False,
            tfm=self.test_transform,
            shuffle=False
        )
        
        self.classnames = few_shot_dataset.classnames
        self.num_classes = len(self.classnames)
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
        
        if 'llm_negatives' in output['aux_losses']:
            # lambda already included in aux_loss computation
            losses['llm_negatives'] = output['aux_losses']['llm_negatives']
        
        if 'mixup_invariance' in output['aux_losses']:
            # lambda already included in aux_loss computation
            losses['mixup_invariance'] = output['aux_losses']['mixup_invariance']
        
        return losses
    
    def train_epoch(self, epoch: int):
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        loss_breakdown = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            # Handle both 2-tuple (legacy) and 3-tuple (with negatives) returns
            if len(batch) == 3:
                images, targets, neg_tokens = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                neg_tokens = neg_tokens.to(self.device) if isinstance(neg_tokens, torch.Tensor) else None
            else:
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                neg_tokens = None
            
            # Forward pass
            if neg_tokens is not None:
                output = self.model(images, targets, negative_text_tokens=neg_tokens)
            else:
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
        """Evaluate on test set and return detailed metrics."""
        self.model.eval()
        
        correct = 0
        total = 0
        total_loss = 0.0
        loss_breakdown = {}
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluation"):
                # Handle both 2-tuple and 3-tuple batch formats
                if len(batch) == 3:
                    images, targets, _ = batch
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                else:
                    images, targets = batch
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                
                # Forward pass
                output = self.model(images, targets)
                
                # Compute losses
                losses = self.compute_losses(output, targets)
                total_loss_val = sum(losses.values())
                
                # Track loss breakdown
                for k, v in losses.items():
                    if k not in loss_breakdown:
                        loss_breakdown[k] = 0.0
                    loss_breakdown[k] += v.item()
                
                # Metrics
                acc = cls_acc(output['logits'], targets, topk=1)
                correct += acc / 100 * len(targets)
                total += len(targets)
                total_loss += total_loss_val.item()
                
                # Collect predictions and targets for detailed analysis
                all_preds.append(output['logits'].detach().cpu())
                all_targets.append(targets.cpu())
        
        test_acc = correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        # Compute per-class accuracy if possible
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        per_class_acc = self._compute_per_class_accuracy(all_preds, all_targets)
        
        return {
            'acc': test_acc,
            'loss': avg_loss,
            'loss_breakdown': loss_breakdown,
            'per_class_acc': per_class_acc,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def train(self, num_epochs: int):
        """Main training loop with per-epoch evaluation."""
        best_acc = 0.0
        best_epoch = 0
        
        # Initialize evaluation history (for JSON logging)
        eval_history = []
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            eval_metrics = self.evaluate()
            test_acc = eval_metrics['acc']
            test_loss = eval_metrics['loss']
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.4f}")
            print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}")
            
            # Print per-class accuracies
            if eval_metrics['per_class_acc']:
                print(f"  Per-class accuracies:")
                for class_name, acc in eval_metrics['per_class_acc'].items():
                    print(f"    {class_name}: {acc:.4f}")
            
            # Print loss breakdown if available
            if eval_metrics['loss_breakdown']:
                print(f"  Test Loss Breakdown:")
                for k, v in eval_metrics['loss_breakdown'].items():
                    print(f"    {k}: {v / len(self.test_loader):.4f}")
            
            # Record to history
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['acc'],
                'test_loss': test_loss,
                'test_acc': test_acc,
                'per_class_acc': eval_metrics['per_class_acc']
            }
            eval_history.append(epoch_record)
            
            # Save best checkpoint
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                self.save_checkpoint()
                print(f"  ✓ New best accuracy! Checkpoint saved.")
            
            # Save attention visualizations periodically
            if (epoch + 1) % self.cfg.get('save_vis_interval', 5) == 0:
                try:
                    self.save_attention_maps(epoch)
                except Exception as e:
                    print(f"Warning: Failed to save attention visualizations: {e}")
        
        # Save evaluation history to JSON
        self._save_eval_history(eval_history)
        
        print(f"\n✓ Training complete.")
        print(f"  Best test accuracy: {best_acc:.4f} (Epoch {best_epoch})")
        return best_acc
    
    def _save_eval_history(self, history: list):
        """Save evaluation history to JSON file."""
        history_path = os.path.join(self.cfg['cache_dir'], 'eval_history.json')
        try:
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"✓ Evaluation history saved to {history_path}")
        except Exception as e:
            print(f"Warning: failed to save eval history: {e}")
        
    def _compute_per_class_accuracy(self, preds, targets):
        """Compute accuracy per class."""
        preds_label = preds.argmax(dim=1)
        per_class_acc = {}
        
        for class_idx in range(self.num_classes):
            mask = targets == class_idx
            if mask.sum() == 0:
                continue
            class_correct = (preds_label[mask] == class_idx).float().mean().item()
            per_class_acc[self.classnames[class_idx]] = class_correct
        
        return per_class_acc
        
        print(f"\n✓ Training complete. Best test accuracy: {best_acc:.4f}")
        return best_acc
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        os.makedirs(self.cfg['cache_dir'], exist_ok=True)
        checkpoint_path = os.path.join(self.cfg['cache_dir'], 'model_best.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    def save_attention_maps(self, epoch: int):
        """
        Visualize and save attention maps (Top-K Binary Mask or Slot Attention Map).
        Shows Input Image with generated mask overlay as PNG.
        
        Args:
            epoch: current epoch number
        """
        import torchvision.transforms as transforms
        from PIL import Image as PILImage
        import numpy as np
        
        self.model.eval()
        
        # Create visualization directory
        vis_dir = os.path.join(self.cfg['cache_dir'], 'attention_maps', f'epoch_{epoch}')
        os.makedirs(vis_dir, exist_ok=True)
        
        print(f"\n✓ Generating attention visualizations for epoch {epoch}...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx >= 5:  # Save only 5 batches to avoid too many files
                    break
                
                # Unpack batch
                if len(batch) == 3:
                    images, targets, neg_tokens = batch
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                else:
                    images, targets = batch
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                
                # Forward pass to get selected features
                try:
                    output = self.model(images, targets)
                except:
                    output = self.model(images)
                
                selected_feats = output['selected_feats']  # (B, K, D)
                B, K, D = selected_feats.shape
                
                # Re-encode images to get local features for mask generation
                with torch.no_grad():
                    if hasattr(self.model, 'module'):
                        image_encoder = self.model.module.image_encoder
                    else:
                        image_encoder = self.model.image_encoder
                    
                    _, local_features = image_encoder(images.type(self.model.module.dtype if hasattr(self.model, 'module') else self.model.dtype))
                
                # Compute importance scores from selected features
                # Map K selected patches back to spatial grid
                N = local_features.shape[1]  # Total number of patches (typically 196 for 14x14)
                grid_size = int(np.sqrt(N))
                
                # Generate masks for each sample in batch
                for b in range(min(B, 3)):  # Save at most 3 samples per batch
                    # Original image (denormalize)
                    img_orig = images[b].cpu().numpy()  # (3, H, W)
                    # Denormalize
                    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
                    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
                    img_orig = (img_orig * std + mean).clip(0, 1)
                    img_orig = (img_orig.transpose(1, 2, 0) * 255).astype(np.uint8)
                    
                    # Generate attention mask
                    # Compute importance from selected features
                    importance = (selected_feats[b] ** 2).sum(dim=-1).cpu().numpy()  # (K,)
                    
                    # Create binary mask of top-K patches
                    mask = np.zeros(N)
                    
                    # For slot attention: use weighted average
                    if K <= N:
                        # Normalize importance scores
                        top_k_importance = np.argsort(-importance)[:K]
                        mask[top_k_importance] = 1.0
                    
                    # Reshape mask to spatial grid
                    mask_grid = mask.reshape(grid_size, grid_size)
                    mask_grid = np.repeat(np.repeat(mask_grid, 224 // grid_size, axis=0), 224 // grid_size, axis=1)
                    mask_grid = mask_grid[:224, :224]  # Ensure exact size
                    
                    # Create heatmap visualization
                    heatmap = (mask_grid * 255).astype(np.uint8)
                    heatmap_colored = np.zeros((224, 224, 3), dtype=np.uint8)
                    heatmap_colored[:, :, 0] = heatmap  # Red channel
                    
                    # Overlay mask on image
                    alpha = 0.5
                    overlay = (img_orig.astype(float) * (1 - alpha) + 
                              heatmap_colored.astype(float) * alpha).astype(np.uint8)
                    
                    # Create side-by-side comparison
                    comparison = np.concatenate([img_orig, overlay], axis=1)
                    
                    # Save image
                    classname = self.classnames[targets[b].item()]
                    filename = f"batch{batch_idx:02d}_sample{b}_class{classname}.png"
                    filepath = os.path.join(vis_dir, filename)
                    
                    pil_img = PILImage.fromarray(comparison)
                    pil_img.save(filepath)
                    
                    print(f"  Saved: {filepath}")
        
        print(f"✓ Attention visualizations saved to {vis_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/my_config.yaml')
    parser.add_argument('--is_train', type=int, default=1)
    parser.add_argument('--cache_root', type=str, default='./my_caches', 
                       help='Root directory for saving outputs (logs, checkpoints, visualizations)')
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
        args.cache_root,  # ← 使用命令行参数指定的根目录
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
