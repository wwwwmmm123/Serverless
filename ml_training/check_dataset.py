"""
检查 Azure 数据集是否有 NaN 或异常值
"""

import json
import numpy as np
from pathlib import Path

def check_dataset(dataset_path: str):
    """检查数据集中的异常值"""
    
    print("="*60)
    print("Dataset Quality Check")
    print("="*60)
    
    nan_count = 0
    inf_count = 0
    sample_count = 0
    
    feature_mins = [float('inf')] * 8
    feature_maxs = [float('-inf')] * 8
    
    print(f"\nLoading and checking: {dataset_path}")
    
    with open(dataset_path) as f:
        for line_num, line in enumerate(f, 1):
            sample = json.loads(line)
            sample_count += 1
            
            # 检查缓存条目
            for entry in sample['cache_entries']:
                features = [
                    entry['cpu'],
                    entry['memory'],
                    entry['cold_start_time'],
                    entry['output_size'],
                    entry['call_frequency'],
                    entry['last_access_time'] / 10000.0,
                    entry['access_count'] / 100.0,
                    1.0 if entry['is_dag_root'] else 0.0,
                ]
                
                # 检查 NaN
                for i, val in enumerate(features):
                    if np.isnan(val):
                        nan_count += 1
                        print(f"[WARN] Line {line_num}, NaN in feature {i}: {entry}")
                        break
                    
                    if np.isinf(val):
                        inf_count += 1
                        print(f"[WARN] Line {line_num}, Inf in feature {i}: {entry}")
                        break
                    
                    # 更新 min/max
                    feature_mins[i] = min(feature_mins[i], val)
                    feature_maxs[i] = max(feature_maxs[i], val)
            
            # 每 10000 个样本报告一次
            if line_num % 10000 == 0:
                print(f"  Checked {line_num:,} samples...")
    
    print(f"\n{'='*60}")
    print("Check Results")
    print(f"{'='*60}")
    print(f"Total samples: {sample_count:,}")
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")
    
    print(f"\nFeature ranges:")
    feature_names = ['cpu', 'memory', 'cold_start', 'output_size', 
                     'call_freq', 'last_access', 'access_count', 'is_dag_root']
    
    for i, name in enumerate(feature_names):
        print(f"  {name:15s}: [{feature_mins[i]:.6f}, {feature_maxs[i]:.6f}]")
    
    if nan_count > 0 or inf_count > 0:
        print(f"\n[ERROR] Dataset contains {nan_count} NaN and {inf_count} Inf values!")
        print("This will cause training to fail.")
        return False
    else:
        print(f"\n[OK] Dataset is clean - no NaN or Inf values")
        return True


if __name__ == "__main__":
    import sys
    
    dataset_path = "datasets/azure_real_cache_decisions.jsonl"
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    is_clean = check_dataset(dataset_path)
    
    if not is_clean:
        print("\nPlease fix the data generation script.")
        sys.exit(1)
    else:
        print("\nDataset is ready for training!")
        sys.exit(0)
