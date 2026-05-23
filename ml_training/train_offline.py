"""
离线训练 Attention-LRU 模型

使用预生成的数据集进行训练。

训练 / 评估惯例（与常见深度学习教程一致）：
- 参数更新：`train_epoch` 内 `model.train()`，开启 Dropout（AttentionLRU 内多处 Dropout + MHA dropout）。
  本管线**无**图像式输入增强；batch 内打印的 Loss/Acc 是「含 Dropout」的监视值，不宜与验证集直接对比。
- 验证集：`evaluate(..., val_loader)`，`model.eval()`，关闭 Dropout。
- 训练集公平指标：每个 epoch 在更新完权重后，再用 `evaluate(..., train_loader)`、`model.eval()` 扫一遍训练集，
  得到与验证同一度量下的训练损失/准确率；**曲线与 Epoch 摘要中的 Train 行均使用该值**。
- 摘要里另有一行 `Train-step (dropout on)`，仅反映优化过程内的随机前向，可选对照。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import argparse

from attention_lru import AttentionLRU


class CacheDecisionDataset(Dataset):
    """缓存决策数据集"""
    
    def __init__(self, jsonl_path: str):
        self.samples = []
        
        print(f"Loading dataset from {jsonl_path}...")
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        print(f"[OK] Loaded {len(self.samples):,} samples")
        self.feature_dim = 8
        if self.samples:
            e0 = self.samples[0]["cache_entries"][0]
            if "downstream_count" in e0:
                self.feature_dim = 12
        print(f"[OK] feature_dim={self.feature_dim} (8=Azure/合成, 12=Alibaba DAG)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        返回:
            features: (cache_size, feature_dim) 的特征矩阵
            can_be_evicted: (cache_size,) 的布尔掩码
            label: 最优驱逐索引
        """
        sample = self.samples[idx]
        
        # 提取特征
        cache_entries = sample['cache_entries']
        features = []
        
        for entry in cache_entries:
            feature_vec = [
                entry['cpu'],
                entry['memory'],
                entry['cold_start_time'],
                entry['output_size'],
                entry['call_frequency'],
                min(entry['last_access_time'] / 10000.0, 10.0),
                min(entry['access_count'] / 100.0, 10.0),
                1.0 if entry['is_dag_root'] else 0.0,
            ]
            if self.feature_dim == 12:
                feature_vec.extend(
                    [
                        min(entry.get("downstream_count", 0) / 10.0, 10.0),
                        min(entry.get("upstream_count", 0) / 10.0, 10.0),
                        min(entry.get("dag_depth", 0) / 10.0, 10.0),
                        1.0 if entry.get("downstream_count", 0) > 0 else 0.0,
                    ]
                )
            features.append(feature_vec)
        
        features = torch.tensor(features, dtype=torch.float32)
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features.clamp_(-50.0, 50.0)
        n_slots = features.shape[0]
        ce = torch.tensor(sample['can_be_evicted'], dtype=torch.bool)
        if ce.numel() < n_slots:
            pad = torch.zeros(n_slots - ce.numel(), dtype=torch.bool)
            can_be_evicted = torch.cat([ce, pad], dim=0)
        elif ce.numel() > n_slots:
            can_be_evicted = ce[:n_slots]
        else:
            can_be_evicted = ce
        label = int(sample['optimal_evict_idx'])
        label = max(0, min(label, n_slots - 1))
        
        return features, can_be_evicted, label


def collate_fn(batch):
    """
    Padding 与两种 mask 分离：
    - key_padding_mask: True = padding 槽位，传给 MultiheadAttention（勿与 can_be_evicted 混用）
    - can_be_evicted: 业务上是否允许驱逐（padding 处为 False）
    """
    features_list, masks_list, labels_list = zip(*batch)

    max_len = max(f.shape[0] for f in features_list)
    batch_size = len(batch)
    feature_dim = features_list[0].shape[1]

    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    can_be_evicted = torch.zeros(batch_size, max_len, dtype=torch.bool)

    label_rows: List[int] = []
    for i, (features, evict_mask) in enumerate(zip(features_list, masks_list)):
        seq_len = features.shape[0]
        padded_features[i, :seq_len] = features
        key_padding_mask[i, :seq_len] = False
        can_be_evicted[i, :seq_len] = evict_mask
        li = int(labels_list[i])
        li = max(0, min(li, seq_len - 1))
        label_rows.append(li)

    labels = torch.tensor(label_rows, dtype=torch.long)

    return padded_features, key_padding_mask, can_be_evicted, labels


def train_epoch(model: AttentionLRU, 
                dataloader: DataLoader, 
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device) -> Tuple[float, float]:
    """训练一个 epoch：model.train()，用于反向传播；返回值为 train 模式下的平均 loss/acc（含 Dropout）。"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    valid_batches = 0
    
    for batch_idx, (features, key_padding_mask, can_be_evicted, labels) in enumerate(dataloader):
        features = features.to(device)
        key_padding_mask = key_padding_mask.to(device)
        can_be_evicted = can_be_evicted.to(device)
        labels = labels.to(device)
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features.clamp_(-50.0, 50.0)

        optimizer.zero_grad()
        scores = model(
            features,
            key_padding_mask=key_padding_mask,
            can_be_evicted=can_be_evicted,
        )
        
        # 检查 NaN
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            if batch_idx < 3:
                print(f"[WARN] NaN/Inf in forward at batch {batch_idx} (check JSONL / 重新跑 process_cluster_trace_v2017)")
            continue
        
        # 计算损失
        loss = criterion(scores, labels)
        
        # 检查 loss 是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            if batch_idx < 3:
                print(f"[WARN] Invalid loss at batch {batch_idx}: {loss.item()}")
            continue
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（更激进）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        valid_batches += 1
        
        # 计算准确率
        _, predicted = scores.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 进度显示（降低频率 + flush，避免终端缓冲导致重复观感）
        if (batch_idx + 1) % 50 == 0 and total > 0:
            avg_loss = total_loss / max(valid_batches, 1)
            acc = correct / total
            print(
                f"  Batch {batch_idx+1}/{len(dataloader)}, "
                f"Loss: {avg_loss:.4f}, Acc: {acc:.3f}, Valid: {valid_batches}",
                flush=True,
            )
    
    if valid_batches == 0 or total == 0:
        print("[ERR] 本 epoch 无有效 batch（输入或模型持续 NaN）。请重新生成数据集或降低 lr。")
        return float("nan"), 0.0
    
    avg_loss = total_loss / valid_batches
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model: AttentionLRU,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """model.eval() + no_grad，用于验证集或与验证可比的训练集再评估（关闭 Dropout）。"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    valid_batches = 0
    with torch.no_grad():
        for features, key_padding_mask, can_be_evicted, labels in dataloader:
            features = features.to(device)
            key_padding_mask = key_padding_mask.to(device)
            can_be_evicted = can_be_evicted.to(device)
            labels = labels.to(device)
            features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
            features.clamp_(-50.0, 50.0)

            scores = model(
                features,
                key_padding_mask=key_padding_mask,
                can_be_evicted=can_be_evicted,
            )
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                continue
            loss = criterion(scores, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            total_loss += loss.item()
            valid_batches += 1
            
            _, predicted = scores.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if valid_batches == 0 or total == 0:
        return float("nan"), 0.0
    
    avg_loss = total_loss / valid_batches
    accuracy = correct / total
    
    return avg_loss, accuracy


def plot_training_curves(train_losses: List[float],
                         train_accs: List[float],
                         val_losses: List[float],
                         val_accs: List[float],
                         save_path: str = 'models/offline_training_curves.png'):
    """绘制训练曲线（Train 与 Val 均在 model.eval() 下统计，可与教科书形态对照）"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss 曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss (eval)', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss (eval)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss (train & val both under eval / no dropout)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy 曲线
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc (eval)', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc (eval)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy (train & val both under eval / no dropout)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline training for Attention-LRU")
    parser.add_argument('--dataset', type=str, 
                        default='datasets/alibaba_dag_cache_decisions.jsonl',
                        help='Path to dataset (JSONL format)')
    parser.add_argument(
        '--feature_dim',
        type=int,
        default=None,
        help='Override input feature dim (default: infer from JSONL, 8 or 12)',
    )
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5; attention trains more stably lower)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension of the model')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--save_path', type=str, default='models/attention_lru_offline.pth',
                        help='Model save path')
    parser.add_argument(
        '--plot_path',
        type=str,
        default=None,
        help='Where to save training curves PNG (default: same stem as --save_path + _curves.png)',
    )
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split and train DataLoader shuffle',
    )
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    plot_path = args.plot_path
    if plot_path is None:
        sp = Path(args.save_path)
        plot_path = str(sp.with_name(f"{sp.stem}_curves.png"))
    
    print(f"{'='*60}")
    print(f"Attention-LRU Offline Training")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Hidden dim: {args.hidden_dim}")
    print(f"  - Num heads: {args.num_heads}")
    print(f"  - Val split: {args.val_split}")
    print(f"  - Save path: {args.save_path}")
    print(f"  - Plot path: {plot_path}")
    print(f"  - Device: {args.device}")
    print(f"  - Seed: {args.seed}")
    print(f"{'='*60}\n")
    print(
        "[Note] AttentionLRU 训练时使用 Dropout；若用 train 模式下的 batch 损失与验证对比，\n"
        "       验证集会系统性更优。曲线与 Epoch 摘要中的 Train 指标均在 eval 模式\n"
        "       下对训练集再算一遍，与 Val 公平可比。\n"
    )
    
    # 加载数据集
    full_dataset = CacheDecisionDataset(args.dataset)
    feature_dim = args.feature_dim if args.feature_dim is not None else full_dataset.feature_dim
    if feature_dim != full_dataset.feature_dim:
        print(f"[WARN] --feature_dim={feature_dim} 与数据推断 {full_dataset.feature_dim} 不一致，请确认")
    print(f"[OK] 使用 feature_dim={feature_dim} 训练")
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    split_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=split_gen
    )
    
    print(f"\nDataset split:")
    print(f"  - Train: {len(train_dataset):,} samples")
    print(f"  - Val: {len(val_dataset):,} samples\n")
    
    # 创建 DataLoader
    loader_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows 上设为 0
        generator=loader_gen,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 创建模型
    device = torch.device(args.device)
    model = AttentionLRU(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads
    ).to(device)
    
    print(f"Model created:")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Model size: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024:.1f} KB\n")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练循环
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_loss_step, train_acc_step = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        if train_loss_step != train_loss_step:  # NaN
            print("[ERR] Train loss is NaN; stopping.")
            break

        # 与验证集同一度量：eval 模式 + 无 Dropout，避免「验证长期优于训练」的假信号
        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        if train_loss != train_loss:
            print("[ERR] Train eval loss is NaN; stopping.")
            break

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度（无效 val_loss 时跳过，避免 ReduceLROnPlateau 行为异常）
        if val_loss == val_loss:
            scheduler.step(val_loss)
        
        # 打印统计
        print(f"\nEpoch {epoch} Summary:")
        print(
            f"  Train (eval, 与 Val 可比): Loss {train_loss:.4f}, Acc {train_acc:.3f}"
        )
        print(f"  Val:                       Loss {val_loss:.4f}, Acc {val_acc:.3f}")
        print(
            f"  Train-step (dropout on):   Loss {train_loss_step:.4f}, Acc {train_acc_step:.3f}"
        )
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_loss_step': train_loss_step,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'feature_dim': feature_dim,
            }, args.save_path)
            print(f"  [OK] Best model saved (Val Acc: {val_acc:.3f})")
        
        print()
    
    elapsed_time = time.time() - start_time
    
    # 最终总结
    print(f"{'='*60}")
    print(f"Training Completed")
    print(f"{'='*60}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {args.save_path}")
    print(f"{'='*60}\n")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path=plot_path)


if __name__ == "__main__":
    main()
