"""
Attention-Based LRU Cache Policy
轻量级的基于注意力机制的缓存策略

使用方式:
1. 在线训练模式: 边运行仿真边训练
2. 离线推理模式: 使用训练好的模型进行推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import deque
import json
import os


class AttentionLRU(nn.Module):
    """
    基于注意力机制的 LRU 缓存策略
    
    特点:
    - 轻量级: 只有 ~50K 参数
    - 快速推理: CPU 上 <5ms
    - 易训练: 只需少量数据
    """
    
    def __init__(
        self,
        feature_dim: int = 8,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 1. 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 4. Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 5. 驱逐分数头
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重（略保守，利于稳定训练）"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.67)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        can_be_evicted: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            features: [batch_size, num_functions, feature_dim] 或 [num_functions, feature_dim]
            key_padding_mask: [batch_size, num_functions]，与 nn.MultiheadAttention 一致：
                True 表示该位置为 padding，不参与 attention。
            can_be_evicted: [batch_size, num_functions]，True 表示允许被驱逐。
                为 False 的槽位 logits 会被压到极小。

        注意：`key_padding_mask` 的语义与 ``nn.MultiheadAttention`` 一致（True = 忽略）。
        ``OnlineTrainer`` 传入的 batch 第二参即为 padding mask。
        """
        # 处理单样本情况
        squeeze_output = False
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, N, F]
            squeeze_output = True
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)
            if can_be_evicted is not None:
                can_be_evicted = can_be_evicted.unsqueeze(0)

        batch_size, num_fns, _ = features.shape

        # 1. 特征编码
        hidden = self.feature_encoder(features)  # [B, N, H]

        # 2. Self-Attention（只用 padding mask，勿把 can_be_evicted 当作 key_padding_mask）
        attn_out, attn_weights = self.attention(
            hidden, hidden, hidden,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        hidden = self.norm1(hidden + attn_out)  # [B, N, H]

        # 3. Feed-Forward (with residual connection)
        ffn_out = self.ffn(hidden)
        hidden = self.norm2(hidden + ffn_out)  # [B, N, H]

        # 4. 驱逐分数（logits；CrossEntropy 前不必 softmax）
        scores = self.score_head(hidden).squeeze(-1)  # [B, N]

        # 数值稳定：padding / 不可驱逐槽位压到极小 logit（避免 CE 用 inf）
        mask_fill = -1e4
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, mask_fill)
        if can_be_evicted is not None:
            scores = scores.masked_fill(~can_be_evicted, mask_fill)

        if squeeze_output:
            scores = scores.squeeze(0)  # [N]

        return scores
    
    def select_eviction(
        self,
        features: torch.Tensor,
        can_be_evicted: torch.Tensor
    ) -> int:
        """
        选择应该被驱逐的函数
        
        Args:
            features: [num_functions, feature_dim] 函数特征
            can_be_evicted: [num_functions] bool 数组，哪些可以被驱逐
        
        Returns:
            evict_idx: 应该被驱逐的函数索引
        """
        with torch.no_grad():
            scores = self.forward(features)  # [N]
            
            # 不可驱逐的函数设为无穷大
            scores = scores.clone()
            scores[~can_be_evicted] = float('inf')
            
            # 选择分数最低的
            evict_idx = torch.argmin(scores).item()
        
        return evict_idx
    
    def get_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）
        
        Args:
            features: [num_functions, feature_dim]
        
        Returns:
            attention_weights: [num_functions, num_functions]
        """
        with torch.no_grad():
            if features.dim() == 2:
                features = features.unsqueeze(0)
            
            hidden = self.feature_encoder(features)
            _, attn_weights = self.attention(
                hidden, hidden, hidden,
                need_weights=True,
                average_attn_weights=True
            )
            
            return attn_weights.squeeze(0)  # [N, N]


def extract_features(cache_state: Dict, env_state: Dict) -> Tuple[torch.Tensor, List[int]]:
    """
    从缓存状态提取特征
    
    Args:
        cache_state: {
            'cached_fns': [fn_5, fn_12, fn_33, ...],
            'fn_info': {
                fn_5: {'cpu': 10, 'memory': 500, ...},
                ...
            },
            'access_info': {
                fn_5: {'last_access': 100, 'access_count': 20, ...},
                ...
            }
        }
        env_state: {
            'current_frame': 105,
            'memory_pressure': 0.7,
            ...
        }
    
    Returns:
        features: [num_cached_fns, feature_dim]
        fn_ids: [fn_5, fn_12, fn_33, ...]
    """
    cached_fns = cache_state['cached_fns']
    fn_info = cache_state['fn_info']
    access_info = cache_state['access_info']
    current_frame = env_state['current_frame']
    
    features_list = []
    
    for fn_id in cached_fns:
        info = fn_info[fn_id]
        access = access_info[fn_id]
        
        # 提取 8 维特征
        feat = [
            info.get('cpu', 0.0) / 100.0,                          # 归一化 CPU
            info.get('memory', 0.0) / 1000.0,                      # 归一化内存
            info.get('cold_start_time', 0.0) / 500.0,              # 归一化冷启动
            info.get('output_size', 0.0) / 100.0,                  # 归一化输出大小
            access.get('call_frequency', 0.0),                      # 调用频率 [0, 1]
            (current_frame - access.get('last_access', 0)) / 100.0,  # 归一化时间
            access.get('access_count', 0) / 100.0,                 # 归一化访问次数
            float(info.get('is_dag_root', False))                  # DAG 根节点 {0, 1}
        ]
        
        features_list.append(feat)
    
    features = torch.tensor(features_list, dtype=torch.float32)
    
    return features, cached_fns


class OnlineTrainer:
    """
    在线训练器
    
    边运行仿真边训练模型
    """
    
    def __init__(
        self,
        model: AttentionLRU,
        learning_rate: float = 1e-3,
        buffer_size: int = 10000,
        batch_size: int = 32,
        train_every: int = 100
    ):
        """
        Args:
            model: AttentionLRU 模型
            learning_rate: 学习率
            buffer_size: 经验回放缓冲区大小
            batch_size: 批量大小
            train_every: 每多少步训练一次
        """
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # 经验回放缓冲区
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.train_every = train_every
        self.step_count = 0
        
        # 训练统计
        self.train_losses = []
        self.train_accuracies = []
    
    def add_experience(
        self,
        features: torch.Tensor,
        optimal_evict_idx: int,
        reward: float = None
    ):
        """
        添加一条经验到缓冲区
        
        Args:
            features: [num_functions, feature_dim]
            optimal_evict_idx: 最优策略选择的驱逐索引
            reward: 可选的奖励信号
        """
        self.buffer.append({
            'features': features.clone(),
            'label': optimal_evict_idx,
            'reward': reward
        })
        
        self.step_count += 1
        
        # 每 train_every 步训练一次
        if self.step_count % self.train_every == 0 and len(self.buffer) >= self.batch_size:
            self.train_step()
    
    def train_step(self):
        """执行一次训练步骤"""
        # 从缓冲区采样
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # 准备批量数据
        max_len = max(exp['features'].shape[0] for exp in batch)
        
        batch_features = []
        batch_labels = []
        batch_masks = []
        
        for exp in batch:
            feat = exp['features']
            label = exp['label']
            
            # Padding
            pad_len = max_len - feat.shape[0]
            if pad_len > 0:
                feat = torch.cat([feat, torch.zeros(pad_len, feat.shape[1])], dim=0)
                mask = torch.cat([
                    torch.zeros(feat.shape[0] - pad_len, dtype=torch.bool),
                    torch.ones(pad_len, dtype=torch.bool)
                ])
            else:
                mask = torch.zeros(feat.shape[0], dtype=torch.bool)
            
            batch_features.append(feat)
            batch_labels.append(label)
            batch_masks.append(mask)
        
        batch_features = torch.stack(batch_features)  # [B, max_len, F]
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)  # [B]
        batch_masks = torch.stack(batch_masks)  # [B, max_len]
        
        # 前向传播
        scores = self.model(batch_features, batch_masks)  # [B, max_len]
        
        # 计算损失
        loss = self.criterion(scores, batch_labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 计算准确率
        with torch.no_grad():
            pred = torch.argmax(scores, dim=1)
            accuracy = (pred == batch_labels).float().mean().item()
        
        # 记录统计
        self.train_losses.append(loss.item())
        self.train_accuracies.append(accuracy)
        
        # 定期打印
        if len(self.train_losses) % 10 == 0:
            avg_loss = np.mean(self.train_losses[-10:])
            avg_acc = np.mean(self.train_accuracies[-10:])
            print(f"[Train] Step {self.step_count}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")
    
    def save_model(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'step_count': self.step_count
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.step_count = checkpoint.get('step_count', 0)
        print(f"Model loaded from {path}")


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建模型
    model = AttentionLRU(
        feature_dim=8,
        hidden_dim=64,
        num_heads=4
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024:.1f} KB")
    
    # 测试前向传播
    test_features = torch.randn(5, 8)  # 5 个函数，每个 8 维特征
    scores = model(test_features)
    print(f"\nTest forward pass:")
    print(f"Input shape: {test_features.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Scores: {scores}")
    
    # 测试驱逐选择
    can_evict = torch.tensor([True, False, True, True, False])
    evict_idx = model.select_eviction(test_features, can_evict)
    print(f"\nEviction decision:")
    print(f"Can be evicted: {can_evict}")
    print(f"Selected index: {evict_idx}")
