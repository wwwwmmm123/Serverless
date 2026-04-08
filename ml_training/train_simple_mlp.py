"""
简化版 Attention-LRU 训练脚本（更稳定）

移除复杂的 attention 机制，使用简单的 MLP
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import argparse


class SimpleCacheMLP(nn.Module):
    """简化的缓存策略模型（纯 MLP）"""
    
    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, 1)
        )
        
        # 使用更保守的初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: [batch_size, num_functions, feature_dim]
        
        Returns:
            scores: [batch_size, num_functions] 驱逐分数
        """
        batch_size, num_fns, feature_dim = features.shape
        
        # 重塑为 [batch_size * num_fns, feature_dim]
        features_flat = features.view(-1, feature_dim)
        
        # 通过网络
        scores_flat = self.network(features_flat)  # [B*N, 1]
        
        # 重塑回 [batch_size, num_fns]
        scores = scores_flat.view(batch_size, num_fns)
        
        return scores


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
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """
        返回:
            features: (num_functions, feature_dim)
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
                min(entry['last_access_time'] / 10000.0, 10.0),  # 限制范围
                min(entry['access_count'] / 100.0, 10.0),  # 限制范围
                1.0 if entry['is_dag_root'] else 0.0,
            ]
            features.append(feature_vec)
        
        features = torch.tensor(features, dtype=torch.float32)
        label = sample['optimal_evict_idx']
        
        return features, label


def collate_fn(batch):
    """批处理函数"""
    features_list, labels_list = zip(*batch)
    
    # Padding to same length
    max_len = max(f.shape[0] for f in features_list)
    batch_size = len(batch)
    feature_dim = features_list[0].shape[1]
    
    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    
    for i, features in enumerate(features_list):
        seq_len = features.shape[0]
        padded_features[i, :seq_len] = features
    
    labels = torch.tensor(labels_list, dtype=torch.long)
    
    return padded_features, labels


def plot_training_curves(
    train_losses: List[float],
    train_accs: List[float],
    val_losses: List[float],
    val_accs: List[float],
    save_path: str,
) -> None:
    """Save train/val loss and accuracy vs epoch (same layout as train_offline.py)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, val_accs, "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Training curves saved to {save_path}")


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    valid_batches = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        
        # 检查输入
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"[SKIP] Batch {batch_idx}: invalid input")
            continue
        
        optimizer.zero_grad()
        
        try:
            scores = model(features)
            
            # 检查输出
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(f"[SKIP] Batch {batch_idx}: invalid output")
                continue
            
            loss = criterion(scores, labels)
            
            # 检查损失
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[SKIP] Batch {batch_idx}: invalid loss")
                continue
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            valid_batches += 1
            
            _, predicted = scores.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 进度显示
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / max(valid_batches, 1)
                acc = correct / max(total, 1)
                print(f"  Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {avg_loss:.4f}, Acc: {acc:.3f}, Valid: {valid_batches}/{batch_idx+1}")
        
        except Exception as e:
            print(f"[ERROR] Batch {batch_idx}: {e}")
            continue
    
    if valid_batches == 0:
        return float('nan'), 0.0
    
    avg_loss = total_loss / valid_batches
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    valid_batches = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            try:
                scores = model(features)
                loss = criterion(scores, labels)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    _, predicted = scores.max(1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            except:
                continue
    
    if valid_batches == 0:
        return float('nan'), 0.0
    
    avg_loss = total_loss / valid_batches
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Stable offline training")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)  # 更小的学习率
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--save_path', type=str, default='models/simple_mlp.pth')
    parser.add_argument(
        '--plot_path',
        type=str,
        default='models/simple_mlp_training_curves.png',
        help='Where to save loss/accuracy curves PNG',
    )
    parser.add_argument(
        '--summary_path',
        type=str,
        default='models/simple_mlp_last_run.json',
        help='JSON summary of this run (for comparing with previous training)',
    )
    parser.add_argument('--no_plot', action='store_true', help='Skip saving the curves PNG')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Simple MLP Cache Policy Training (Stable Version)")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Hidden dim: {args.hidden_dim}")
    print("="*60)
    
    # 加载数据
    full_dataset = CacheDecisionDataset(args.dataset)
    
    # 划分数据集
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\nDataset split:")
    print(f"  - Train: {len(train_dataset):,}")
    print(f"  - Val: {len(val_dataset):,}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCacheMLP(feature_dim=8, hidden_dim=args.hidden_dim).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: Simple MLP")
    print(f"  - Parameters: {params:,}")
    print(f"  - Device: {device}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if np.isnan(train_loss):
            print("[ERROR] Training loss is NaN - stopping")
            break
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, args.save_path)
            print(f"  [OK] Best model saved (Val Acc: {val_acc:.3f})")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("Training Completed")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Best Val Acc: {best_val_acc:.3f}")
    print(f"Model saved: {args.save_path}")
    print("="*60)

    if train_losses and not args.no_plot:
        plot_training_curves(
            train_losses, train_accs, val_losses, val_accs, args.plot_path
        )

    summary = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "train_simple_mlp.py",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "seconds": round(elapsed, 2),
        "best_val_acc": best_val_acc,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_train_acc": train_accs[-1] if train_accs else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "final_val_acc": val_accs[-1] if val_accs else None,
        "model_path": args.save_path,
        "plot_path": None if args.no_plot else args.plot_path,
    }
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Run summary saved to {args.summary_path}")


if __name__ == "__main__":
    main()
