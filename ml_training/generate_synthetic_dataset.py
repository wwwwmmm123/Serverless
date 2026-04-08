"""
生成合成缓存决策数据集

基于真实的函数特征分布，合成大量训练数据
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import random


class FunctionProfile:
    """函数特征配置"""
    def __init__(self, fn_type: str):
        self.fn_type = fn_type
        
        # 基于 serverless_sim 的实际配置
        if fn_type == "cpu_intensive":
            self.cpu = (0.7, 0.95)  # (min, max)
            self.memory = (0.1, 0.3)
            self.cold_start = (0.3, 0.6)
            self.output_size = (0.1, 0.2)
        elif fn_type == "data_intensive":
            self.cpu = (0.1, 0.3)
            self.memory = (0.6, 0.9)
            self.cold_start = (0.2, 0.4)
            self.output_size = (0.7, 0.9)
        elif fn_type == "memory_intensive":
            self.cpu = (0.2, 0.4)
            self.memory = (0.7, 0.95)
            self.cold_start = (0.4, 0.7)
            self.output_size = (0.1, 0.3)
        else:  # lightweight
            self.cpu = (0.05, 0.2)
            self.memory = (0.05, 0.2)
            self.cold_start = (0.05, 0.15)
            self.output_size = (0.05, 0.15)


def generate_function_features(fn_type: str, fn_id: int, current_time: int) -> Dict:
    """生成一个函数的特征"""
    profile = FunctionProfile(fn_type)
    
    return {
        'fn_id': fn_id,
        'cpu': np.random.uniform(*profile.cpu),
        'memory': np.random.uniform(*profile.memory),
        'cold_start_time': np.random.uniform(*profile.cold_start),
        'output_size': np.random.uniform(*profile.output_size),
        'call_frequency': np.random.exponential(0.3),  # 指数分布
        'last_access_time': current_time - np.random.randint(0, 100),
        'access_count': np.random.randint(1, 50),
        'is_dag_root': np.random.random() < 0.2,  # 20% 是 DAG 根节点
    }


def calculate_optimal_eviction_lru(cache_entries: List[Dict]) -> int:
    """LRU 策略：驱逐最久未使用的"""
    return min(enumerate(cache_entries), 
               key=lambda x: x[1]['last_access_time'])[0]


def calculate_optimal_eviction_env_aware(cache_entries: List[Dict], 
                                          current_time: int) -> int:
    """EnvAwareLRU 策略（作为教师）"""
    def score(entry):
        # 参考 env_aware_lru.rs 的评分公式
        cold_weight = 0.3
        mem_weight = 0.2
        freq_weight = 0.3
        recency_weight = 0.2
        
        recency = (current_time - entry['last_access_time']) / max(current_time, 1)
        frequency = entry['call_frequency']
        cold_start = entry['cold_start_time']
        memory = entry['memory']
        
        return (cold_weight * cold_start + 
                mem_weight * memory + 
                freq_weight * (1.0 / (frequency + 1e-6)) + 
                recency_weight * recency)
    
    return max(enumerate(cache_entries), key=lambda x: score(x[1]))[0]


def generate_cache_decision(decision_id: int, 
                            cache_capacity: int = 10,
                            current_time: int = 0) -> Dict:
    """生成一个缓存决策样本"""
    
    # 随机选择函数类型分布
    fn_types = ["cpu_intensive", "data_intensive", "memory_intensive", "lightweight"]
    type_weights = [0.3, 0.3, 0.2, 0.2]
    
    # 生成缓存中的函数（容量已满）
    cache_entries = []
    for i in range(cache_capacity):
        fn_type = np.random.choice(fn_types, p=type_weights)
        cache_entries.append(generate_function_features(fn_type, i, current_time))
    
    # 生成新到达的函数
    new_fn_type = np.random.choice(fn_types, p=type_weights)
    new_fn = generate_function_features(new_fn_type, cache_capacity, current_time)
    
    # 计算最优驱逐（使用 EnvAwareLRU 作为教师）
    optimal_evict_idx = calculate_optimal_eviction_env_aware(cache_entries, current_time)
    
    return {
        'decision_id': decision_id,
        'timestamp': current_time,
        'cache_entries': cache_entries,
        'new_fn': new_fn,
        'optimal_evict_idx': optimal_evict_idx,
        'can_be_evicted': [True] * cache_capacity,  # 假设都可驱逐
    }


def generate_dataset(num_samples: int = 10000, 
                     output_path: str = "datasets/synthetic_cache_decisions.jsonl") -> None:
    """
    生成合成数据集
    
    Args:
        num_samples: 样本数量
        output_path: 输出路径
    """
    
    print(f"{'='*60}")
    print(f"Generating Synthetic Cache Decision Dataset")
    print(f"{'='*60}")
    print(f"Number of samples: {num_samples:,}")
    print(f"Output path: {output_path}")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 生成数据集
    with open(output_path, 'w') as f:
        for i in range(num_samples):
            # 模拟时间推进
            current_time = i * 10
            
            decision = generate_cache_decision(i, cache_capacity=10, current_time=current_time)
            
            # 写入 JSONL 格式
            f.write(json.dumps(decision) + '\n')
            
            # 进度显示
            if (i + 1) % 1000 == 0:
                print(f"Generated {i+1:,}/{num_samples:,} samples ({(i+1)/num_samples*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset generated successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")


def analyze_dataset(dataset_path: str) -> None:
    """分析数据集统计信息"""
    print(f"\n{'='*60}")
    print(f"Dataset Analysis")
    print(f"{'='*60}")
    
    # 统计信息
    eviction_distribution = {}
    fn_type_counts = {'cpu_intensive': 0, 'data_intensive': 0, 
                      'memory_intensive': 0, 'lightweight': 0}
    
    sample_count = 0
    with open(dataset_path) as f:
        for line in f:
            sample = json.loads(line)
            sample_count += 1
            
            # 驱逐位置分布
            evict_idx = sample['optimal_evict_idx']
            eviction_distribution[evict_idx] = eviction_distribution.get(evict_idx, 0) + 1
    
    print(f"Total samples: {sample_count:,}")
    print(f"\nEviction index distribution:")
    for idx in sorted(eviction_distribution.keys()):
        count = eviction_distribution[idx]
        pct = count / sample_count * 100
        print(f"  Position {idx}: {count:>5,} ({pct:>5.2f}%)")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic cache decision dataset")
    parser.add_argument('--num_samples', type=int, default=10000, 
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--output', type=str, default='datasets/synthetic_cache_decisions.jsonl',
                        help='Output file path')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze the generated dataset')
    
    args = parser.parse_args()
    
    # 生成数据集
    generate_dataset(args.num_samples, args.output)
    
    # 分析（如果需要）
    if args.analyze:
        analyze_dataset(args.output)
