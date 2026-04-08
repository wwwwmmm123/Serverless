"""
在线训练脚本 - 与 batch_run.py 集成

工作流程:
1. 运行 batch_run.py，使用 env_aware_lru 作为教师策略
2. 拦截每次驱逐决策，收集数据
3. 边运行边训练 Attention-LRU 模型
4. 保存训练好的模型

使用方式:
    python train_online.py --episodes 1000 --save_path models/attention_lru.pth
"""

import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(os.path.dirname(__file__))

import torch
import argparse
from attention_lru import AttentionLRU, OnlineTrainer, extract_features
from proxy_env3 import ProxyEnv3
import time
import json
from typing import Dict, List
import matplotlib.pyplot as plt


class TrainingCollector:
    """
    训练数据收集器
    
    拦截仿真过程，收集缓存决策数据
    """
    
    def __init__(self, env: ProxyEnv3, trainer: OnlineTrainer):
        self.env = env
        self.trainer = trainer
        self.episode_count = 0
        self.decision_count = 0
        
    def run_episode(self, max_steps: int = 1000):
        """
        运行一个 episode，收集训练数据
        
        Args:
            max_steps: 最大步数
        """
        print(f"\n{'='*60}")
        print(f"Episode {self.episode_count + 1}")
        print(f"{'='*60}")
        
        # 重置环境
        try:
            kernel = self.env.reset()
            print(f"Environment reset successful, env_id: {self.env.env_id}")
        except Exception as e:
            print(f"ERROR: Failed to reset environment: {e}")
            return False
        
        # 运行仿真
        for step in range(max_steps):
            try:
                # 执行一步
                result = self.env.step(1)
                
                # 这里我们需要从 result 中提取缓存状态
                # 但当前 API 可能不直接提供，所以我们先做简单版本
                # 实际使用时需要修改 Rust 代码暴露缓存状态
                
                if step % 100 == 0:
                    print(f"  Step {step}/{max_steps}")
                
            except Exception as e:
                print(f"ERROR at step {step}: {e}")
                break
        
        self.episode_count += 1
        print(f"Episode {self.episode_count} completed")
        
        return True
    
    def collect_cache_decision(
        self,
        cache_state: Dict,
        env_state: Dict,
        optimal_evict_fn: int
    ):
        """
        收集一次缓存驱逐决策
        
        Args:
            cache_state: 当前缓存状态
            env_state: 当前环境状态
            optimal_evict_fn: 教师策略选择的驱逐函数
        """
        try:
            # 提取特征
            features, fn_ids = extract_features(cache_state, env_state)
            
            # 找到 optimal_evict_fn 的索引
            if optimal_evict_fn in fn_ids:
                label = fn_ids.index(optimal_evict_fn)
            else:
                print(f"WARNING: optimal_evict_fn {optimal_evict_fn} not in fn_ids")
                return
            
            # 添加到训练器
            self.trainer.add_experience(features, label)
            
            self.decision_count += 1
            
        except Exception as e:
            print(f"ERROR in collect_cache_decision: {e}")


def plot_training_progress(trainer: OnlineTrainer, save_path: str):
    """绘制训练进度"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(trainer.train_losses)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(trainer.train_accuracies)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Online training for Attention-LRU')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to run (default: 100)')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Max steps per episode (default: 1000)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--train_every', type=int, default=100,
                       help='Train every N experiences (default: 100)')
    parser.add_argument('--save_path', type=str, default='models/attention_lru.pth',
                       help='Model save path (default: models/attention_lru.pth)')
    parser.add_argument('--plot_path', type=str, default='models/training_plot.png',
                       help='Training plot save path')
    
    args = parser.parse_args()
    
    print(f"""
{'='*60}
Attention-LRU Online Training
{'='*60}
Configuration:
  - Episodes: {args.episodes}
  - Max steps per episode: {args.max_steps}
  - Learning rate: {args.learning_rate}
  - Batch size: {args.batch_size}
  - Train every: {args.train_every} experiences
  - Save path: {args.save_path}
{'='*60}
""")
    
    # 1. 创建模型和训练器
    model = AttentionLRU(
        feature_dim=8,
        hidden_dim=64,
        num_heads=4
    )
    
    trainer = OnlineTrainer(
        model=model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_every=args.train_every
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created:")
    print(f"  - Parameters: {total_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024:.1f} KB")
    print()
    
    # 2. 连接到仿真环境
    print("Connecting to simulation environment...")
    try:
        env = ProxyEnv3()
        print("✓ Connected to simulation environment")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nPlease make sure the Rust server is running:")
        print("  cd d:\\serverless\\serverless_sim\\serverless_sim")
        print("  cargo run --release")
        return
    
    # 3. 创建数据收集器
    collector = TrainingCollector(env, trainer)
    
    # 4. 运行训练
    print(f"\nStarting training for {args.episodes} episodes...")
    start_time = time.time()
    
    successful_episodes = 0
    for episode in range(args.episodes):
        success = collector.run_episode(args.max_steps)
        if success:
            successful_episodes += 1
        
        # 定期保存模型
        if (episode + 1) % 10 == 0:
            trainer.save_model(args.save_path)
            print(f"Model checkpoint saved ({episode + 1}/{args.episodes})")
    
    elapsed_time = time.time() - start_time
    
    # 5. 保存最终模型
    trainer.save_model(args.save_path)
    
    # 6. 绘制训练曲线
    if len(trainer.train_losses) > 0:
        plot_training_progress(trainer, args.plot_path)
    
    # 7. 打印总结
    avg_loss_last_100 = np.mean(trainer.train_losses[-100:]) if len(trainer.train_losses) >= 100 else float('nan')
    avg_acc_last_100 = np.mean(trainer.train_accuracies[-100:]) if len(trainer.train_accuracies) >= 100 else float('nan')
    
    print(f"""
{'='*60}
Training Completed
{'='*60}
Statistics:
  - Total episodes: {args.episodes}
  - Successful episodes: {successful_episodes}
  - Total decisions: {collector.decision_count}
  - Training steps: {len(trainer.train_losses)}
  - Time elapsed: {elapsed_time:.1f} seconds
  
Model saved to: {args.save_path}

Final performance:
  - Avg loss (last 100): {avg_loss_last_100:.4f if not np.isnan(avg_loss_last_100) else 'N/A'}
  - Avg accuracy (last 100): {avg_acc_last_100:.3f if not np.isnan(avg_acc_last_100) else 'N/A'}
{'='*60}
""")


if __name__ == "__main__":
    import numpy as np  # 添加缺失的 import
    main()
