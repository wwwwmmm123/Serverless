"""
分析数据集的标签分布
"""

import json
import numpy as np
from collections import Counter

def analyze_labels(dataset_path: str):
    """分析标签分布"""
    
    print("="*60)
    print("Label Distribution Analysis")
    print("="*60)
    
    labels = []
    
    with open(dataset_path) as f:
        for line in f:
            sample = json.loads(line)
            labels.append(sample['optimal_evict_idx'])
    
    label_counts = Counter(labels)
    total = len(labels)
    
    print(f"\nTotal samples: {total:,}")
    print(f"Unique labels: {len(label_counts)}")
    print(f"\nLabel distribution:")
    
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  Position {label}: {count:>6,} ({pct:>5.2f}%) {bar}")
    
    # 计算熵（衡量标签的均匀性）
    probs = np.array([label_counts[i] for i in range(10)]) / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(10)  # 10 个类别的最大熵
    
    print(f"\nLabel entropy: {entropy:.3f} / {max_entropy:.3f} ({entropy/max_entropy*100:.1f}%)")
    
    if entropy / max_entropy < 0.5:
        print("[WARN] Labels are highly imbalanced!")
        print("This may cause the model to overfit to majority class.")
    elif entropy / max_entropy > 0.9:
        print("[OK] Labels are well balanced.")
    else:
        print("[OK] Labels have moderate balance.")


if __name__ == "__main__":
    analyze_labels("datasets/azure_real_cache_decisions.jsonl")
