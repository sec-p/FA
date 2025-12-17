import os
import sys
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets

from utils import *
import clip
from model_v2 import SimpleCLIP

from my_dataset import build_dataset
from my_dataset.utils import build_data_loader

from ood_utils.ood_tool import get_measures


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/my_config.yaml", help='settings in yaml format')
    parser.add_argument('--is_train', type=int, default=1, help='1->train 0->test')
    parser.add_argument('--use_full_data', type=int, default=0, help='1->use full dataset, 0->use few-shot')
    args = parser.parse_args()
    return args


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"GPU num: {num_gpus}")

    # Load config
    args = get_arguments()
    assert os.path.exists(args.config)
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['device'] = device
    cfg['is_train'] = args.is_train
    cfg['use_full_data'] = args.use_full_data

    # Load CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model = torch.nn.DataParallel(clip_model).to(device)
    clip_model = clip_model.module

    model_file_name_dict = {
        "RN50": "RN50",
        "RN101": "RN101",
        "RN50x4": "RN50x4",
        "RN50x16": "RN50x16",
        "ViT-B/32": "ViT-B-32",
        "ViT-B/16": "ViT-B-16",
    }

    model_path = os.path.join(cfg['clip_model_path'], model_file_name_dict[cfg['backbone']] + ".pt")
    model_temp = torch.jit.load(model_path, map_location=device).eval()
    state_dict = model_temp.state_dict()
    cfg['embed_dim'] = state_dict["text_projection"].shape[1]

    K_name = str(cfg['K']).replace('.', '-')
    lr_name = str(cfg['lr']).replace('.', '')
    
    data_mode = 'fulldata' if cfg['use_full_data'] else str(cfg['shots']) + 'shots'
    model_name = 'SimpleCLIP_v2'
    cache_dir = os.path.join(
        './my_caches', cfg['id_dataset'], model_file_name_dict[cfg['backbone']],
        data_mode, model_name + '_batch' + str(cfg['fine_tune_batch_size']) +
        '_ep' + str(cfg['fine_tune_train_epoch']), 'K-' + str(K_name),
        'lr' + lr_name, 'seed' + str(cfg['seed'])
    )

    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    # Seed
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # Transforms
    train_transform_aug = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    train_transform_no_aug = preprocess

    if cfg['is_train'] == 1:
        sys.stdout = Logger(os.path.join(cache_dir, 'log_train.txt'), stream=sys.stdout)
        print("\nRunning configs.")
        print(cfg, "\n")

        # Dataset
        few_shot_dataset = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'] if not cfg['use_full_data'] else -1)
        batch_size = cfg['fine_tune_batch_size']

        if cfg['use_full_data']:
            # Use full dataset
            train_loader = build_data_loader(
                data_source=few_shot_dataset.train_x + few_shot_dataset.val,  # Combine train + val
                batch_size=batch_size, tfm=train_transform_aug, is_train=True, shuffle=True
            )
            test_loader = build_data_loader(
                data_source=few_shot_dataset.test, batch_size=batch_size,
                is_train=False, tfm=train_transform_no_aug, shuffle=False
            )
        else:
            # Use few-shot setup
            train_loader = build_data_loader(
                data_source=few_shot_dataset.train_x, batch_size=batch_size,
                tfm=train_transform_aug, is_train=True, shuffle=True
            )
            test_loader = build_data_loader(
                data_source=few_shot_dataset.test, batch_size=batch_size,
                is_train=False, tfm=train_transform_no_aug, shuffle=False
            )

        cfg['classnames'] = few_shot_dataset.classnames

        print(f"Data mode: {'Full Dataset' if cfg['use_full_data'] else 'Few-shot (' + str(cfg['shots']) + '-shot)'}")
        print(f"Classnames: {cfg['classnames']}")
        print(f"Num classes: {len(cfg['classnames'])}")
        print(f"Use RL: {cfg.get('use_rl', False)}")

        # Model
        model = SimpleCLIP(cfg, cfg['classnames'], clip_model)
        model = torch.nn.DataParallel(model).to(device)

        # Only gating is trainable
        for name, param in model.named_parameters():
            if "gating" in name:
                param.requires_grad_(True)
                print(f"trainable: {name}")
            else:
                param.requires_grad_(False)

        # Optimizer
        gate_params = [p for n, p in model.named_parameters() if p.requires_grad]
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(gate_params, lr=cfg.get('lr_gate', cfg['lr'] * 0.1), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg['fine_tune_train_epoch'] * len(train_loader)
        )

        print('Optimizer:', optimizer)
        print('Scheduler:', scheduler)

        train_epoch = cfg['fine_tune_train_epoch']

        for train_idx in range(train_epoch):
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))

            model.train()

            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()

                logits, logits_local, aux_loss, rl_info = model(images, target)
                loss_ce = criterion(logits, target)

                if not cfg.get('use_rl', False):
                    # Soft gating mode
                    sparsity = aux_loss.get('sparsity', torch.tensor(0.0, device=images.device))
                    minimal = aux_loss.get('minimal', torch.tensor(0.0, device=images.device))
                    total_loss = loss_ce + cfg.get('lambda_sparsity', 1e-3) * sparsity + cfg.get('lambda_minimal', 1e-2) * minimal

                    acc = cls_acc(output=logits, target=target, topk=1)
                    correct_samples += acc / 100 * len(logits)
                    all_samples += len(logits)
                    loss_list.append(total_loss.item())

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()
                else:
                    # RL mode
                    acc = cls_acc(output=logits, target=target, topk=1)
                    correct_samples += acc / 100 * len(logits)
                    all_samples += len(logits)
                    loss_list.append(loss_ce.item())

                    # Encoder update (only CE, but gating is not part of encoder here)
                    optimizer.zero_grad()
                    loss_ce.backward()
                    optimizer.step()
                    scheduler.step()

                    # Policy update (gating)
                    with torch.no_grad():
                        preds = logits.argmax(dim=1)
                        base_reward = torch.where(
                            preds == target,
                            torch.ones_like(preds, dtype=torch.float32),
                            -1 * torch.ones_like(preds, dtype=torch.float32)
                        ).float().to(images.device)

                    mask_ratio = rl_info.get('mask_ratio', torch.zeros(images.size(0), device=images.device))
                    confusion = rl_info.get('confusion_penalty', torch.tensor(0.0, device=images.device))
                    lambda_sp = cfg.get('lambda_sparsity', 1e-3)
                    lambda_min = cfg.get('lambda_minimal', 1e-2)

                    if mask_ratio.dim() == 0:
                        mask_ratio = mask_ratio.repeat(images.size(0))

                    rewards = base_reward - lambda_sp * mask_ratio - lambda_min * confusion

                    baseline = model.module.baseline if isinstance(model, torch.nn.DataParallel) else model.baseline
                    advantage = rewards - baseline

                    log_prob = rl_info.get('log_prob', torch.zeros(images.size(0), device=images.device))
                    policy_loss = -(log_prob * advantage).mean()

                    optimizer.zero_grad()
                    policy_loss.backward()
                    optimizer.step()

                    # Update baseline
                    new_baseline = 0.9 * baseline + 0.1 * rewards.mean()
                    if isinstance(model, torch.nn.DataParallel):
                        model.module.baseline.copy_(new_baseline.detach())
                    else:
                        model.baseline.copy_(new_baseline.detach())

                    loss_list.append(policy_loss.item())

            current_lr = scheduler.get_last_lr()[0]
            avg_loss = sum(loss_list) / len(loss_list) if len(loss_list) > 0 else 0.0
            train_acc = correct_samples / all_samples if all_samples > 0 else 0.0
            print('LR: {:.6f}, train_acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(
                current_lr, train_acc, int(correct_samples), int(all_samples), avg_loss
            ))

            # Test
            if train_idx == train_epoch - 1:
                model.eval()
                with torch.no_grad():
                    correct_samples, all_samples = 0, 0
                    topk = 1

                    for images, labels in tqdm(test_loader):
                        images, labels = images.to(device), labels.to(device)
                        outputs, _, _, _ = model(images)

                        pred = outputs.topk(topk, 1, True, True)[1].t()
                        correct = pred.eq(labels.view(1, -1).expand_as(pred))
                        acc_num = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                        correct_samples += acc_num
                        all_samples += labels.shape[0]

                    test_acc = correct_samples / all_samples
                    print(f'Test Epoch [{train_idx+1}/{train_epoch}], Test Accuracy: {100 * test_acc:.2f}%')

                    # Save model
                    torch.save(model.state_dict(), os.path.join(cfg['cache_dir'], 'model.pth'))

    else:
        # Evaluation mode (OOD detection)
        test_file_name = f'log_test_ood_{cfg["ood_dataset"]}.txt'
        sys.stdout = Logger(os.path.join(cache_dir, test_file_name), stream=sys.stdout)
        print("\nRunning OOD Evaluation configs.")
        print(cfg, "\n")

        few_shot_dataset = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'] if not cfg['use_full_data'] else -1)
        batch_size = cfg['test_batch_size']

        test_loader_id = build_data_loader(
            data_source=few_shot_dataset.test, batch_size=batch_size,
            is_train=False, tfm=preprocess, shuffle=False
        )

        cfg['classnames'] = few_shot_dataset.classnames

        print(f"Num classes: {len(cfg['classnames'])}")

        # Model
        model = SimpleCLIP(cfg, cfg['classnames'], clip_model)
        model = torch.nn.DataParallel(model).to(device)

        # Load model
        model.load_state_dict(torch.load(os.path.join(cfg['cache_dir'], 'model.pth')))

        model.eval()
        with torch.no_grad():
            correct_samples, all_samples = 0, 0
            topk = 1
            T = 1.0

            id_conf = np.array([])

            # Run ID dataset
            for images, labels in tqdm(test_loader_id):
                images, labels = images.to(device), labels.to(device)

                logits_id, logits_id_local, _, _ = model(images)
                logits_id /= 100.0

                smax_global = F.softmax(logits_id / T, dim=-1).cpu().numpy()
                mcm_global_score = np.max(smax_global, axis=1)

                id_conf = np.concatenate((id_conf, mcm_global_score))

                pred = logits_id.topk(topk, 1, True, True)[1].t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                acc_num = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                correct_samples += acc_num
                all_samples += labels.shape[0]

            test_acc = correct_samples / all_samples
            print(f'ID Test Accuracy: {100 * test_acc:.2f}%')

            # OOD dataset handling
            ood_dataset_name = cfg['ood_dataset']
            if ood_dataset_name == 'challenging':
                ood_dataset_name = ['OpenImage_O', 'NINCO', 'imagenet-o']
            elif ood_dataset_name == 'all':
                ood_dataset_name = ['iNaturalist', 'SUN', 'Places', 'dtd', 'OpenImage_O', 'NINCO', 'imagenet-o']
            elif ood_dataset_name == 'nearood':
                ood_dataset_name = ['ssb_hard', 'NINCO']
            elif ood_dataset_name == 'common':
                ood_dataset_name = ['iNaturalist', 'SUN', 'Places', 'dtd']
            else:
                ood_dataset_name = [ood_dataset_name]

            print(f"OOD datasets: {ood_dataset_name}")

            avg_auroc, avg_aupr, avg_fpr = 0.0, 0.0, 0.0

            for cur_ood_dataset in ood_dataset_name:
                print(f'Evaluating OOD dataset: {cur_ood_dataset}')
                ood_conf = np.array([])

                if cur_ood_dataset == 'ood':
                    ood_dataset = datasets.ImageFolder(
                        root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset), transform=preprocess
                    )
                elif cur_ood_dataset in ['iNaturalist', 'SUN', 'Places', 'OpenImage_O', 'imagenet-o', 'ssb_hard']:
                    ood_dataset = datasets.ImageFolder(
                        root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset), transform=preprocess
                    )
                elif cur_ood_dataset == 'dtd':
                    ood_dataset = datasets.ImageFolder(
                        root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'images'), transform=preprocess
                    )
                elif cur_ood_dataset == 'NINCO':
                    ood_dataset = datasets.ImageFolder(
                        root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'NINCO_OOD_classes'),
                        transform=preprocess
                    )
                else:
                    ood_dataset = datasets.ImageFolder(
                        root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'test'), transform=preprocess
                    )

                ood_loader = torch.utils.data.DataLoader(
                    ood_dataset, batch_size=cfg['test_batch_size'], shuffle=False, num_workers=8
                )

                for images, _ in tqdm(ood_loader):
                    images = images.to(device)
                    logits_ood, _, _, _ = model(images)
                    logits_ood /= 100.0

                    smax_global = F.softmax(logits_ood / T, dim=-1).cpu().numpy()
                    mcm_global_score = np.max(smax_global, axis=1)
                    ood_conf = np.concatenate((ood_conf, mcm_global_score))

                print(f'ID conf size: {id_conf.size}, OOD conf size: {ood_conf.size}')

                auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr=0.95)
                avg_auroc += auroc
                avg_aupr += aupr
                avg_fpr += fpr
                print(f'{cur_ood_dataset}: AUROC={auroc:.4f}, AUPR={aupr:.4f}, FPR95={fpr:.4f}')

            print('=' * 50)
            avg_auroc /= len(ood_dataset_name)
            avg_aupr /= len(ood_dataset_name)
            avg_fpr /= len(ood_dataset_name)
            print(f'Average OOD Performance:')
            print(f'AUROC: {avg_auroc:.4f}, AUPR: {avg_aupr:.4f}, FPR(0.95): {avg_fpr:.4f}')


if __name__ == '__main__':
    main()
