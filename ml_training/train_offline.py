"""
离线训练 Attention-LRU 模型

使用预生成的数据集进行训练
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
            features.append(feature_vec)
        
        features = torch.tensor(features, dtype=torch.float32)
        can_be_evicted = torch.tensor(sample['can_be_evicted'], dtype=torch.bool)
        label = sample['optimal_evict_idx']
        
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

    for i, (features, evict_mask) in enumerate(zip(features_list, masks_list)):
        seq_len = features.shape[0]
        padded_features[i, :seq_len] = features
        key_padding_mask[i, :seq_len] = False
        can_be_evicted[i, :seq_len] = evict_mask

    labels = torch.tensor(labels_list, dtype=torch.long)

    return padded_features, key_padding_mask, can_be_evicted, labels


def train_epoch(model: AttentionLRU, 
                dataloader: DataLoader, 
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device) -> Tuple[float, float]:
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (features, key_padding_mask, can_be_evicted, labels) in enumerate(dataloader):
        features = features.to(device)
        key_padding_mask = key_padding_mask.to(device)
        can_be_evicted = can_be_evicted.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        scores = model(
            features,
            key_padding_mask=key_padding_mask,
            can_be_evicted=can_be_evicted,
        )
        
        # 检查 NaN
        if torch.isnan(scores).any():
            print(f"[WARN] NaN detected in forward pass at batch {batch_idx}")
            continue
        
        # 计算损失
        loss = criterion(scores, labels)
        
        # 检查 loss 是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] Invalid loss at batch {batch_idx}: {loss.item()}")
            continue
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（更激进）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 计算准确率
        _, predicted = scores.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 进度显示
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = correct / total
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, "
                  f"Loss: {avg_loss:.4f}, Acc: {acc:.3f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model: AttentionLRU,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, key_padding_mask, can_be_evicted, labels in dataloader:
            features = features.to(device)
            key_padding_mask = key_padding_mask.to(device)
            can_be_evicted = can_be_evicted.to(device)
            labels = labels.to(device)

            scores = model(
                features,
                key_padding_mask=key_padding_mask,
                can_be_evicted=can_be_evicted,
            )
            loss = criterion(scores, labels)
            
            total_loss += loss.item()
            
            _, predicted = scores.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def plot_training_curves(train_losses: List[float],
                         train_accs: List[float],
                         val_losses: List[float],
                         val_accs: List[float],
                         save_path: str = 'models/offline_training_curves.png'):
    """绘制训练曲线"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss 曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy 曲线
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline training for Attention-LRU")
    parser.add_argument('--dataset', type=str, 
                        default='datasets/synthetic_cache_decisions.jsonl',
                        help='Path to dataset (JSONL format)')
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
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
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
    print(f"  - Device: {args.device}")
    print(f"{'='*60}\n")
    
    # 加载数据集
    full_dataset = CacheDecisionDataset(args.dataset)
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\nDataset split:")
    print(f"  - Train: {len(train_dataset):,} samples")
    print(f"  - Val: {len(val_dataset):,} samples\n")
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows 上设为 0
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
        feature_dim=8,
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
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印统计
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
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
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)


if __name__ == "__main__":
    main()
