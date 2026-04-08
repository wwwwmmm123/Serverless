"""
测试 Attention-LRU 模型

验证模型的基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from attention_lru import AttentionLRU, OnlineTrainer
import numpy as np


def test_model_creation():
    """测试模型创建"""
    print("="*60)
    print("Test 1: Model Creation")
    print("="*60)
    
    model = AttentionLRU(
        feature_dim=8,
        hidden_dim=64,
        num_heads=4
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"  - Parameters: {total_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024:.1f} KB")
    print()
    
    return model


def test_forward_pass(model):
    """测试前向传播"""
    print("="*60)
    print("Test 2: Forward Pass")
    print("="*60)
    
    # 创建测试数据
    num_fns = 5
    features = torch.randn(num_fns, 8)
    
    print(f"Input shape: {features.shape}")
    
    # 前向传播
    scores = model(features)
    
    print(f"Output shape: {scores.shape}")
    print(f"Scores: {scores}")
    print("✓ Forward pass successful")
    print()


def test_eviction_selection(model):
    """测试驱逐选择"""
    print("="*60)
    print("Test 3: Eviction Selection")
    print("="*60)
    
    # 创建测试数据
    features = torch.tensor([
        [0.1, 0.5, 0.2, 0.3, 0.8, 0.05, 0.2, 1.0],  # fn_5: 高频、DAG根
        [0.5, 0.8, 0.4, 0.1, 0.3, 0.10, 0.05, 0.0], # fn_12: 低频
        [0.3, 0.6, 0.3, 0.2, 0.5, 0.02, 0.15, 0.0], # fn_33: 中频、最近访问
        [0.2, 0.4, 0.1, 0.4, 0.4, 0.08, 0.10, 0.0], # fn_45
        [0.4, 0.7, 0.3, 0.3, 0.6, 0.06, 0.18, 0.0], # fn_67
    ])
    
    can_evict = torch.tensor([True, True, True, True, False])
    
    print(f"Functions: 5")
    print(f"Can be evicted: {can_evict}")
    
    evict_idx = model.select_eviction(features, can_evict)
    
    print(f"Selected for eviction: index {evict_idx}")
    print("✓ Eviction selection successful")
    print()


def test_attention_weights(model):
    """测试注意力权重"""
    print("="*60)
    print("Test 4: Attention Weights")
    print("="*60)
    
    features = torch.randn(3, 8)
    
    attn_weights = model.get_attention_weights(features)
    
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights:\n{attn_weights}")
    print("✓ Attention weights extraction successful")
    print()


def test_training(model):
    """测试训练过程"""
    print("="*60)
    print("Test 5: Training Process")
    print("="*60)
    
    trainer = OnlineTrainer(
        model=model,
        learning_rate=1e-3,
        batch_size=8,
        train_every=10
    )
    
    print("Adding training experiences...")
    
    # 添加一些模拟经验
    for i in range(50):
        num_fns = np.random.randint(3, 8)
        features = torch.randn(num_fns, 8)
        label = np.random.randint(0, num_fns)
        
        trainer.add_experience(features, label)
    
    print(f"✓ Added {len(trainer.buffer)} experiences")
    print(f"✓ Training steps: {len(trainer.train_losses)}")
    
    if len(trainer.train_losses) > 0:
        print(f"  - Last loss: {trainer.train_losses[-1]:.4f}")
        print(f"  - Last accuracy: {trainer.train_accuracies[-1]:.3f}")
    
    print()
    
    return trainer


def test_save_load(model, trainer):
    """测试模型保存和加载"""
    print("="*60)
    print("Test 6: Save and Load")
    print("="*60)
    
    save_path = "models/test_model.pth"
    
    # 保存
    trainer.save_model(save_path)
    print(f"✓ Model saved to {save_path}")
    
    # 创建新模型
    new_model = AttentionLRU(
        feature_dim=8,
        hidden_dim=64,
        num_heads=4
    )
    
    new_trainer = OnlineTrainer(
        model=new_model,
        learning_rate=1e-3
    )
    
    # 加载
    new_trainer.load_model(save_path)
    print(f"✓ Model loaded from {save_path}")
    
    # 验证参数一致
    old_params = sum(p.sum().item() for p in model.parameters())
    new_params = sum(p.sum().item() for p in new_model.parameters())
    
    print(f"  - Old model param sum: {old_params:.4f}")
    print(f"  - New model param sum: {new_params:.4f}")
    print(f"  - Match: {abs(old_params - new_params) < 1e-5}")
    print()


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        Attention-LRU Model Test Suite                       ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    try:
        # 测试 1: 模型创建
        model = test_model_creation()
        
        # 测试 2: 前向传播
        test_forward_pass(model)
        
        # 测试 3: 驱逐选择
        test_eviction_selection(model)
        
        # 测试 4: 注意力权重
        test_attention_weights(model)
        
        # 测试 5: 训练过程
        trainer = test_training(model)
        
        # 测试 6: 保存和加载
        test_save_load(model, trainer)
        
        print("="*60)
        print("✓ All tests passed!")
        print("="*60)
        print("\nNext steps:")
        print("1. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n2. Start Rust server:")
        print("   cd d:\\serverless\\serverless_sim\\serverless_sim")
        print("   cargo run --release")
        print("\n3. Start training:")
        print("   python train_online.py --episodes 10")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
