"""
处理 Azure Functions 2021 真实数据集并转换为缓存决策训练数据

数据集: AzureFunctionsInvocationTraceForTwoWeeksJan2021
论文: Faster and Cheaper Serverless Computing on Harvested Resources (SOSP 2021)
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import zipfile
import rarfile
from tqdm import tqdm


class AzureTraceProcessor:
    """Azure Functions Trace 处理器"""
    
    def __init__(self, trace_dir: str = r"C:\Users\王盟啊\Downloads\AzurePublicDataset-master\data"):
        self.trace_dir = Path(trace_dir)
        self.rar_file = self.trace_dir / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar"
        
        print(f"Azure Trace Processor initialized")
        print(f"Trace directory: {self.trace_dir}")
        print(f"RAR file: {self.rar_file}")
    
    def extract_rar(self, output_dir: str = None) -> Path:
        """解压 RAR 文件"""
        if output_dir is None:
            output_dir = self.trace_dir / "extracted"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已经解压（支持 .csv 和 .txt 格式）
        csv_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.txt"))
        
        # 也检查 data 目录本身（用户可能直接解压到那里）
        if not csv_files:
            parent_files = list(self.trace_dir.glob("*.csv")) + list(self.trace_dir.glob("*.txt"))
            # 过滤掉很小的文件（可能是元数据）
            parent_files = [f for f in parent_files if f.stat().st_size > 1_000_000]  # > 1MB
            if parent_files:
                # 找到最大的文件（通常是主数据文件）
                csv_files = [max(parent_files, key=lambda f: f.stat().st_size)]
        
        if csv_files:
            print(f"[OK] Found extracted file: {csv_files[0]}")
            print(f"  Size: {csv_files[0].stat().st_size / 1024 / 1024:.2f} MB")
            return csv_files[0]
        
        print(f"\nExtracting {self.rar_file.name}...")
        print("This may take a few minutes...")
        
        # 尝试方法 1: 使用 patool（跨平台，自动查找工具）
        try:
            import patool
            print("Using patool for extraction...")
            patool.extract_archive(str(self.rar_file), outdir=str(output_dir))
            
            csv_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.txt"))
            if csv_files:
                print(f"[OK] Extracted to: {csv_files[0]}")
                return csv_files[0]
        except ImportError:
            print("patool not available, trying alternative methods...")
        except Exception as e:
            print(f"patool extraction failed: {e}")
        
        # 尝试方法 2: 使用 py7zr（纯 Python，支持 RAR）
        try:
            import py7zr
            print("Using py7zr for extraction...")
            with py7zr.SevenZipFile(self.rar_file, mode='r') as archive:
                archive.extractall(path=output_dir)
            
            csv_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.txt"))
            if csv_files:
                print(f"[OK] Extracted to: {csv_files[0]}")
                return csv_files[0]
        except ImportError:
            print("py7zr not available, trying alternative methods...")
        except Exception as e:
            print(f"py7zr extraction failed: {e}")
        
        # 尝试方法 3: 使用 subprocess 调用系统工具
        try:
            import subprocess
            import shutil
            
            # 尝试使用 7-Zip（Windows 常见）
            seven_zip_paths = [
                r"C:\Program Files\7-Zip\7z.exe",
                r"C:\Program Files (x86)\7-Zip\7z.exe",
                shutil.which("7z"),
            ]
            
            for seven_zip in seven_zip_paths:
                if seven_zip and Path(seven_zip).exists():
                    print(f"Using 7-Zip: {seven_zip}")
                    result = subprocess.run(
                        [seven_zip, 'x', str(self.rar_file), f'-o{output_dir}', '-y'],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        csv_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.txt"))
                        if csv_files:
                            print(f"[OK] Extracted to: {csv_files[0]}")
                            return csv_files[0]
                    break
        except Exception as e:
            print(f"Subprocess extraction failed: {e}")
        
        # 所有方法都失败了，给出手动解压指示
        print("\n" + "="*70)
        print("⚠️  自动解压失败")
        print("="*70)
        print("\n请手动解压 RAR 文件：")
        print(f"\n1. 找到文件:")
        print(f"   {self.rar_file}")
        print(f"\n2. 右键 → 解压到此处（或使用 WinRAR/7-Zip）")
        print(f"\n3. 将解压后的 CSV/TXT 文件放到:")
        print(f"   {output_dir}")
        print(f"   或保持在 {self.trace_dir}")
        print(f"\n4. 重新运行脚本:")
        print(f"   python process_azure_real_data.py")
        print("\n" + "="*70)
        print("\n提示: 你也可以安装解压工具:")
        print("  pip install patool")
        print("  或安装 7-Zip: https://www.7-zip.org/")
        print("="*70)
        
        raise FileNotFoundError(
            f"无法自动解压 RAR 文件。请手动解压 {self.rar_file} 到 {output_dir}"
        )
    
    def load_trace(self, csv_path: Path = None, nrows: int = None) -> pd.DataFrame:
        """
        加载 Azure Functions trace
        
        Args:
            csv_path: CSV 文件路径（如果为 None，自动解压）
            nrows: 读取的行数（用于测试，None 表示全部）
        """
        if csv_path is None:
            csv_path = self.extract_rar()
        
        print(f"\nLoading trace from {csv_path}...")
        if nrows:
            print(f"(Reading first {nrows:,} rows for testing)")
        
        # 读取 CSV
        df = pd.read_csv(csv_path, nrows=nrows)
        
        print(f"[OK] Loaded {len(df):,} invocations")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nDataset info:")
        print(f"  - Unique apps: {df['app'].nunique():,}")
        print(f"  - Unique functions: {df.groupby('app')['func'].nunique().sum():,}")
        print(f"  - Time range: {df['end_timestamp'].min():.2f} to {df['end_timestamp'].max():.2f} seconds")
        print(f"  - Total duration: {(df['end_timestamp'].max() - df['end_timestamp'].min()) / 86400:.2f} days")
        print(f"  - Avg invocation duration: {df['duration'].mean():.3f} seconds")
        print(f"  - Median duration: {df['duration'].median():.3f} seconds")
        
        return df
    
    def compute_function_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算每个函数的统计特征
        
        特征包括:
        - 平均/中位数执行时间
        - 执行时间标准差
        - 调用频率
        - 冷启动估计
        """
        print("\nComputing function features...")
        
        # 创建唯一函数 ID（app + func）
        df['func_id'] = df['app'] + '_' + df['func']
        
        # 按时间排序
        df = df.sort_values('end_timestamp').reset_index(drop=True)
        
        # 计算每个函数的统计信息
        func_stats = df.groupby('func_id').agg({
            'duration': ['mean', 'median', 'std', 'min', 'max'],
            'end_timestamp': ['count', 'min', 'max']
        }).reset_index()
        
        func_stats.columns = ['func_id', 'avg_duration', 'median_duration', 'std_duration', 
                             'min_duration', 'max_duration', 'call_count', 
                             'first_invocation', 'last_invocation']
        
        # 计算调用频率（每秒调用次数）
        time_span = func_stats['last_invocation'] - func_stats['first_invocation']
        time_span = time_span.replace(0, 1)  # 避免除零
        func_stats['call_frequency'] = func_stats['call_count'] / time_span
        
        # 估计冷启动比例（简化版本：如果间隔 > 5 分钟，认为是冷启动）
        print("  Estimating cold start ratio...")
        cold_start_counts = []
        
        for func_id in tqdm(func_stats['func_id'], desc="  Processing functions"):
            func_invocations = df[df['func_id'] == func_id]['end_timestamp'].values
            
            if len(func_invocations) <= 1:
                cold_start_counts.append(1)
                continue
            
            # 计算调用间隔
            intervals = np.diff(func_invocations)
            
            # 间隔超过 300 秒（5 分钟）认为是冷启动
            cold_starts = np.sum(intervals > 300) + 1  # +1 for first invocation
            cold_start_counts.append(cold_starts)
        
        func_stats['cold_start_count'] = cold_start_counts
        func_stats['cold_start_ratio'] = func_stats['cold_start_count'] / func_stats['call_count']
        
        # 估计内存使用（根据执行时间和文献中的相关性）
        # 简化假设：内存 ∝ log(duration)
        func_stats['estimated_memory'] = np.log1p(func_stats['avg_duration'] * 1000) * 10  # MB
        
        print(f"[OK] Computed features for {len(func_stats):,} functions")
        print(f"\nFunction statistics:")
        print(f"  - Avg calls per function: {func_stats['call_count'].mean():.1f}")
        print(f"  - Avg cold start ratio: {func_stats['cold_start_ratio'].mean():.2%}")
        print(f"  - Avg call frequency: {func_stats['call_frequency'].mean():.4f} calls/sec")
        
        return func_stats
    
    def normalize_features(self, func_stats: pd.DataFrame) -> pd.DataFrame:
        """归一化特征到 [0, 1] 区间"""
        print("\nNormalizing features...")
        
        normalized = func_stats.copy()
        
        # 需要归一化的列
        columns_to_normalize = [
            'avg_duration', 'median_duration', 'std_duration',
            'call_frequency', 'cold_start_ratio', 'estimated_memory'
        ]
        
        for col in columns_to_normalize:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            
            if max_val > min_val:
                normalized[f'{col}_norm'] = (normalized[col] - min_val) / (max_val - min_val)
            else:
                normalized[f'{col}_norm'] = 0.5
        
        # 对数归一化（用于长尾分布）
        normalized['log_call_count'] = np.log1p(normalized['call_count'])
        normalized['log_call_count_norm'] = (
            (normalized['log_call_count'] - normalized['log_call_count'].min()) /
            (normalized['log_call_count'].max() - normalized['log_call_count'].min())
        )
        
        print("[OK] Features normalized")
        
        return normalized


class CacheDecisionGenerator:
    """从 Azure trace 生成缓存决策数据集"""
    
    def __init__(self, func_stats: pd.DataFrame, cache_capacity: int = 10):
        self.func_stats = func_stats
        self.cache_capacity = cache_capacity
        
        # 选择调用频率较高的函数（更有代表性）
        self.active_functions = func_stats[func_stats['call_count'] >= 10].copy()
        
        print(f"\nCache Decision Generator initialized")
        print(f"  - Total functions: {len(func_stats):,}")
        print(f"  - Active functions (>= 10 calls): {len(self.active_functions):,}")
        print(f"  - Cache capacity: {cache_capacity}")
    
    def generate_decisions(self, num_samples: int = 50000) -> List[Dict]:
        """
        生成缓存决策样本
        
        模拟场景:
        - 缓存已满（包含 cache_capacity 个函数）
        - 新函数到达，需要驱逐一个旧函数
        - 使用 EnvAwareLRU 策略计算最优驱逐
        """
        print(f"\nGenerating {num_samples:,} cache decision samples...")
        
        decisions = []
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            # 随机选择 cache_capacity 个活跃函数作为缓存内容
            if len(self.active_functions) < self.cache_capacity:
                # 如果活跃函数不够，从全部函数中选择
                cached_fns = self.func_stats.sample(n=self.cache_capacity)
            else:
                cached_fns = self.active_functions.sample(n=self.cache_capacity)
            
            # 选择一个新函数（可能来自活跃或不活跃的函数）
            new_fn = self.active_functions.sample(n=1).iloc[0]
            
            # 构建缓存条目特征
            cache_entries = []
            current_time = i * 10  # 模拟时间戳
            
            for idx, (_, fn) in enumerate(cached_fns.iterrows()):
                entry = {
                    'fn_id': idx,
                    
                    # 使用归一化后的特征
                    'cpu': float(fn['avg_duration_norm']),
                    'memory': float(fn['estimated_memory_norm']),
                    'cold_start_time': float(fn['cold_start_ratio_norm'] * fn['avg_duration_norm']),
                    'output_size': float(fn['estimated_memory_norm'] * 0.3),  # 假设输出是内存的30%
                    'call_frequency': float(fn['call_frequency_norm']),
                    
                    # 模拟最后访问时间和访问次数（确保非负）
                    'last_access_time': max(0, current_time - np.random.randint(0, min(current_time + 1, 100))),
                    'access_count': min(int(fn['call_count'] / 10), 100),
                    
                    # 随机分配是否是 DAG 根节点
                    'is_dag_root': bool(np.random.random() < 0.2),
                }
                cache_entries.append(entry)
            
            # 构建新函数特征
            new_fn_entry = {
                'fn_id': self.cache_capacity,
                'cpu': float(new_fn['avg_duration_norm']),
                'memory': float(new_fn['estimated_memory_norm']),
                'cold_start_time': float(new_fn['cold_start_ratio_norm'] * new_fn['avg_duration_norm']),
                'output_size': float(new_fn['estimated_memory_norm'] * 0.3),
                'call_frequency': float(new_fn['call_frequency_norm']),
                'last_access_time': current_time,
                'access_count': 1,
                'is_dag_root': bool(np.random.random() < 0.2),
            }
            
            # 计算最优驱逐（使用 EnvAwareLRU 策略）
            optimal_evict_idx = self._calculate_optimal_eviction(cache_entries, current_time)
            
            decision = {
                'decision_id': i,
                'timestamp': current_time,
                'cache_entries': cache_entries,
                'new_fn': new_fn_entry,
                'optimal_evict_idx': optimal_evict_idx,
                'can_be_evicted': [True] * self.cache_capacity,
            }
            
            decisions.append(decision)
        
        print(f"[OK] Generated {len(decisions):,} decisions")
        
        return decisions
    
    def _calculate_optimal_eviction(self, cache_entries: List[Dict], current_time: int) -> int:
        """使用 EnvAwareLRU 策略计算最优驱逐索引"""
        # EnvAwareLRU 权重（与 Rust 实现保持一致）
        cold_weight = 0.3
        mem_weight = 0.2
        freq_weight = 0.3
        recency_weight = 0.2
        
        def score(entry):
            recency = (current_time - entry['last_access_time']) / max(current_time, 1)
            frequency = entry['call_frequency']
            cold_start = entry['cold_start_time']
            memory = entry['memory']
            
            # 分数越高，越应该被驱逐
            return (cold_weight * cold_start + 
                    mem_weight * memory + 
                    freq_weight * (1.0 / (frequency + 1e-6)) + 
                    recency_weight * recency)
        
        scores = [score(entry) for entry in cache_entries]
        return int(np.argmax(scores))
    
    def save_to_jsonl(self, decisions: List[Dict], output_path: str):
        """保存为 JSONL 格式"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving to {output_path}...")
        
        with open(output_path, 'w') as f:
            for decision in decisions:
                f.write(json.dumps(decision) + '\n')
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        
        print(f"[OK] Saved {len(decisions):,} decisions")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")


def main():
    print("="*70)
    print("Azure Functions 2021 Trace → Cache Decision Dataset Converter")
    print("="*70)
    
    # 步骤 1: 加载 Azure Functions trace
    processor = AzureTraceProcessor()
    
    # 首先用少量数据测试（快速）
    print("\n" + "="*70)
    print("STEP 1: Load Azure Functions Trace")
    print("="*70)
    
    # 加载数据（使用全部数据，如果太慢可以设置 nrows=100000）
    df = processor.load_trace(nrows=None)  # None = 全部数据
    
    # 步骤 2: 计算函数特征
    print("\n" + "="*70)
    print("STEP 2: Compute Function Features")
    print("="*70)
    
    func_stats = processor.compute_function_features(df)
    
    # 步骤 3: 归一化特征
    func_stats_normalized = processor.normalize_features(func_stats)
    
    # 步骤 4: 生成缓存决策数据集
    print("\n" + "="*70)
    print("STEP 3: Generate Cache Decision Dataset")
    print("="*70)
    
    generator = CacheDecisionGenerator(func_stats_normalized, cache_capacity=10)
    decisions = generator.generate_decisions(num_samples=50000)
    
    # 步骤 5: 保存
    output_path = 'datasets/azure_real_cache_decisions.jsonl'
    generator.save_to_jsonl(decisions, output_path)
    
    # 保存函数统计（用于分析）
    stats_path = 'datasets/azure_function_stats.csv'
    func_stats_normalized.to_csv(stats_path, index=False)
    print(f"\n[OK] Saved function statistics to {stats_path}")
    
    # 最终总结
    print("\n" + "="*70)
    print("[OK] Conversion Completed Successfully!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. Training dataset: {output_path}")
    print(f"  2. Function stats: {stats_path}")
    print(f"\nNext steps:")
    print(f"  1. Train model:")
    print(f"     cd d:\\serverless\\serverless_sim\\ml_training")
    print(f"     python train_offline.py --dataset {output_path} --epochs 50 --batch_size 64")
    print(f"\n  2. Compare performance:")
    print(f"     python train_offline.py --dataset datasets/synthetic_cache_decisions.jsonl")
    print(f"     python train_offline.py --dataset {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
