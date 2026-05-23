"""
Alibaba Cluster Trace v2018 -> 缓存决策 JSONL（12 维含启发式 DAG 特征）

与 v2017 的区别：
  - batch_task: 9 列（增加了 task_type 字段）
  - batch_instance: 14 列（schema 有调整）

v2018 Schema（根据实际数据推断）：
  batch_task.csv (9列):
    0: task_name (如 "M1")
    1: task_type (如 "1")
    2: job_name (如 "j_1")
    3: task_id (如 "1")
    4: status (如 "Terminated")
    5: start_time (如 "419912")
    6: end_time (如 "419912")
    7: plan_cpu (如 "100")
    8: plan_mem (如 "0.2")

  batch_instance.csv (14列):
    0: instance_name (如 "ins_74901673")
    1: task_name (如 "task_LTg...")
    2: job_name (如 "j_217")
    3: task_id (如 "10")
    4: status (如 "Terminated")
    5: start_time (如 "673795")
    6: end_time (如 "673797")
    7: machine_id (如 "m_2637")
    8: seq_no (如 "1")
    9: total_seq_no (如 "1")
    10: cpu_?? (如 "13")
    11: mem_?? (如 "16")
    12: real_cpu_max (如 "0.02")
    13: real_mem_max (如 "0.02")

默认数据目录示例：
  .../cluster-trace-v2018/data_mini

用法:
  python process_cluster_trace_v2018.py ^
    --data_dir "D:\\trainData\\cluster-trace-v2018\\data_mini" ^
    --output datasets/cluster2018_batch_cache_decisions.jsonl ^
    --num_samples 50000 ^
    --max_task_rows 800000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _teacher_eviction_score(
    entry: Dict[str, Any],
    current_time: int,
    # 基础4维特征权重（与实验最优教师一致）
    cold_w: float = 0.2,     # 冷启动权重 20%
    mem_w: float = 0.1,      # 内存权重 10%
    freq_w: float = 0.5,     # 频率权重 50%（核心）
    rec_w: float = 0.2,      # 最近性权重 20%
    # DAG拓扑特征权重（额外8维中提取的关键4维）
    dag_root_w: float = 0.1, # DAG根节点：保留（驱逐导致全链冷启动）
    dag_depth_w: float = 0.1, # DAG深度：深度大应保留（级联冷启动代价大）
    downstream_w: float = 0.1, # 下游节点数：下游多应保留（影响范围广）
    upstream_w: float = 0.05,  # 上游节点数：上游少可考虑驱逐
) -> float:
    """
    12维教师策略：分数越高越应该被驱逐。
    
    结合基础4维特征 + DAG拓扑特征（8维中的关键4维），共8维有效特征。
    
    基础逻辑：
    - 冷启动时间长 → 减分（不该驱逐）
    - 内存占用大 → 加分（优先驱逐）
    - 调用频率低 → 加分（优先驱逐）
    - 久未访问 → 加分（优先驱逐）
    
    DAG拓扑逻辑：
    - 根节点 → 减分（驱逐导致全链冷启动）
    - DAG深度大 → 减分（级联冷启动代价高）
    - 下游节点多 → 减分（影响范围广）
    - 上游节点多 → 轻微减分（依赖方多）
    """
    recency = (current_time - entry["last_access_time"]) / max(current_time, 1)
    frequency = entry["call_frequency"]
    cold_start = entry["cold_start_time"]
    memory = entry["memory"]
    
    # DAG拓扑特征
    is_root = 1.0 if entry.get("is_dag_root", False) else 0.0
    dag_depth = min(entry.get("dag_depth", 0) / 10.0, 1.0)  # 归一化
    downstream = min(entry.get("downstream_count", 0) / 10.0, 1.0)
    upstream = min(entry.get("upstream_count", 0) / 10.0, 1.0)
    
    return (
        # 基础4维：与环境感知策略一致
        - cold_w * cold_start
        + mem_w * memory
        + freq_w * (1.0 / (frequency + 1e-6))
        + rec_w * recency
        # DAG拓扑4维：新增的全局感知能力
        - dag_root_w * is_root
        - dag_depth_w * dag_depth
        - downstream_w * downstream
        - upstream_w * upstream
    )


def _load_batch_task_v2018(path: Path, max_rows: int) -> pd.DataFrame:
    """v2018 batch_task: 9列，无表头"""
    cols = [
        "task_name",      # 0: 任务名
        "task_type",      # 1: 任务类型（v2018新增）
        "job_name",       # 2: 作业名（对应v2017的job_id）
        "task_id",        # 3: 任务ID
        "status",         # 4: 状态
        "start_time",     # 5: 开始时间
        "end_time",       # 6: 结束时间
        "plan_cpu",       # 7: CPU计划
        "plan_mem",       # 8: 内存计划
    ]
    df = pd.read_csv(
        path,
        header=None,
        names=cols,
        nrows=max_rows,
        low_memory=False,
    )
    # 数值列转换
    numeric_cols = ["start_time", "end_time", "plan_cpu", "plan_mem"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df["plan_cpu"] = df["plan_cpu"].fillna(0.0).clip(lower=0.0)
    df["plan_mem"] = df["plan_mem"].fillna(0.0).clip(lower=0.0, upper=1.0)
    df = df.dropna(subset=["job_name", "task_id", "start_time"])
    return df


def _load_batch_instance_v2018(path: Path, max_rows: int) -> pd.DataFrame:
    """v2018 batch_instance: 14列，无表头"""
    cols = [
        "instance_name",  # 0: 实例名（v2018新增）
        "task_name",      # 1: 任务名
        "job_name",       # 2: 作业名
        "task_id",        # 3: 任务ID
        "status",         # 4: 状态
        "start_time",     # 5: 开始时间
        "end_time",       # 6: 结束时间
        "machine_id",     # 7: 机器ID
        "seq_no",         # 8: 序列号
        "total_seq_no",   # 9: 总序列号
        "cpu_metric1",    # 10: CPU指标1（用途不明，可能是max/avg的变体）
        "mem_metric1",    # 11: 内存指标1
        "real_cpu_max",   # 12: 实际CPU最大值
        "real_mem_max",   # 13: 实际内存最大值
    ]
    df = pd.read_csv(
        path,
        header=None,
        names=cols,
        nrows=max_rows,
        low_memory=False,
    )
    # 数值列转换
    numeric_cols = ["start_time", "end_time", "seq_no", "total_seq_no", 
                    "real_cpu_max", "real_mem_max", "cpu_metric1", "mem_metric1"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.dropna(subset=["job_name", "task_id"])
    return df


def _duration_stats_v2018(inst: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    """(job_name, task_id) -> 平均运行时长（秒），用于冷启动近似。"""
    df = inst.copy()
    df = df[(df["start_time"] > 0) & (df["end_time"] > df["start_time"])]
    if df.empty:
        return {}
    df["dur"] = df["end_time"] - df["start_time"]
    g = df.groupby(["job_name", "task_id"])["dur"].mean()
    return {tuple(k): float(v) for k, v in g.items()}


def _task_frequency_v2018(inst: pd.DataFrame) -> Dict[Tuple[str, int], int]:
    return (
        inst.groupby(["job_name", "task_id"])
        .size()
        .to_dict()
    )


def build_jobs_from_tasks_v2018(task_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """job_name -> 按 start_time 排序的任务列表（含特征字段）。"""
    jobs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, row in task_df.iterrows():
        job_name = str(row["job_name"])
        task_id = int(row["task_id"])
        jobs[job_name].append(
            {
                "job_name": job_name,
                "task_id": task_id,
                "start_time": int(row["start_time"]),
                "plan_cpu": float(row["plan_cpu"]),
                "plan_mem": float(row["plan_mem"]),
                "task_type": str(row.get("task_type", "")),
            }
        )
    for job_name in jobs:
        jobs[job_name].sort(key=lambda x: (x["start_time"], x["task_id"]))
    return dict(jobs)


def generate_samples_v2018(
    jobs: Dict[str, List[Dict[str, Any]]],
    dur_map: Dict[Tuple[str, int], float],
    freq_map: Dict[Tuple[str, int], int],
    cache_capacity: int,
    num_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    valid_jobs = [jn for jn, tasks in jobs.items() if len(tasks) >= cache_capacity + 1]
    if not valid_jobs:
        raise ValueError(
            "没有 job 的任务数 >= cache_capacity+1。请增大 --max_task_rows 或减小 --cache_capacity。"
        )

    max_freq = max(freq_map.values()) if freq_map else 1
    cpu_vals = [t["plan_cpu"] for ts in jobs.values() for t in ts]
    max_cpu = float(np.nanmax(cpu_vals)) if cpu_vals else 1.0
    if not np.isfinite(max_cpu) or max_cpu <= 0:
        max_cpu = 1.0
    dur_vals = list(dur_map.values()) if dur_map else []
    max_dur = float(np.nanmax(dur_vals)) if dur_vals else 1.0
    if not np.isfinite(max_dur) or max_dur <= 0:
        max_dur = 1.0

    out: List[Dict[str, Any]] = []
    for i in tqdm(range(num_samples), desc="cluster-2018 samples"):
        job_name = rng.choice(valid_jobs)
        tasks = jobs[job_name]
        n = len(tasks)
        chosen_idx = sorted(rng.sample(range(n), cache_capacity))
        current_time = i * 10 + 10_000

        cache_entries: List[Dict[str, Any]] = []
        for slot, idx in enumerate(chosen_idx):
            t = tasks[idx]
            key = (t["job_name"], t["task_id"])
            dur = dur_map.get(key, 30.0)
            freq_n = freq_map.get(key, 1)

            pc = float(t["plan_cpu"]) if np.isfinite(t["plan_cpu"]) else 0.0
            pm = float(t["plan_mem"]) if np.isfinite(t["plan_mem"]) else 0.0
            cpu = float(np.clip(pc / max(max_cpu, 1e-6), 0.0, 1.0))
            memory = float(np.clip(pm, 0.0, 1.0))
            dsafe = float(dur) if np.isfinite(dur) else 30.0
            cold = float(np.clip(dsafe / max(max_dur, 1e-6), 0.0, 1.0))
            call_frequency = float(freq_n / max_freq)
            last_access_time = max(
                0, current_time - rng.randint(0, min(current_time, 500))
            )
            access_count = min(freq_n, 100)

            # 线性链启发式：排序下标 idx
            downstream_count = 1 if idx < n - 1 else 0
            upstream_count = 1 if idx > 0 else 0
            dag_depth = idx
            is_dag_root = idx == 0

            cache_entries.append(
                {
                    "fn_id": slot,
                    "job_name": job_name,
                    "task_id": t["task_id"],
                    "cpu": cpu,
                    "memory": memory,
                    "cold_start_time": cold,
                    "output_size": float(np.clip(memory * 0.3, 0.0, 1.0)),
                    "call_frequency": call_frequency,
                    "last_access_time": last_access_time,
                    "access_count": access_count,
                    "is_dag_root": is_dag_root,
                    "downstream_count": downstream_count,
                    "upstream_count": upstream_count,
                    "dag_depth": dag_depth,
                }
            )

        scores = [_teacher_eviction_score(e, current_time) for e in cache_entries]
        optimal_evict_idx = int(np.argmax(scores))

        out.append(
            {
                "decision_id": i,
                "dag_source": "alibaba_cluster_2018_batch_chain_heuristic",
                "job_name_sample": job_name,
                "timestamp": current_time,
                "cache_entries": cache_entries,
                "optimal_evict_idx": optimal_evict_idx,
                "can_be_evicted": [True] * cache_capacity,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster trace v2018 -> cache JSONL")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="cluster-trace-v2018 目录（含 batch_task.csv, batch_instance.csv）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/cluster2018_batch_cache_decisions.jsonl",
    )
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--cache_capacity", type=int, default=10)
    parser.add_argument(
        "--max_task_rows",
        type=int,
        default=800_000,
        help="batch_task / batch_instance 各自最多读取行数",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir 不存在: {data_dir}")

    task_path = data_dir / "batch_task.csv"
    inst_path = data_dir / "batch_instance.csv"
    if not task_path.is_file():
        raise SystemExit(f"未找到 {task_path}")
    if not inst_path.is_file():
        raise SystemExit(f"未找到 {inst_path}（需要用于时长/频次统计）")

    print(f"[OK] 读取 batch_task.csv (v2018 schema, 最多 {args.max_task_rows:,} 行)...")
    task_df = _load_batch_task_v2018(task_path, args.max_task_rows)
    print(f"[OK] 读取 batch_instance.csv (v2018 schema, 最多 {args.max_task_rows:,} 行)...")
    inst_df = _load_batch_instance_v2018(inst_path, args.max_task_rows)

    print("[OK] 构建 job -> tasks ...")
    jobs = build_jobs_from_tasks_v2018(task_df)
    print(f"[OK] job 数: {len(jobs):,}")

    dur_map = _duration_stats_v2018(inst_df)
    freq_map = _task_frequency_v2018(inst_df)
    print(f"[OK] 时长键: {len(dur_map):,}, 频次键: {len(freq_map):,}")

    decisions = generate_samples_v2018(
        jobs,
        dur_map,
        freq_map,
        cache_capacity=args.cache_capacity,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] 写入 {len(decisions):,} 行 -> {out_path}")
    print("[OK] 训练示例:")
    print(
        f"  python train_simple_mlp.py --dataset {out_path.name} "
        f"--save_path models/simple_mlp_cluster2018.pth"
    )
    print(
        f"  python train_offline.py --dataset {out_path.name} "
        f"--save_path models/attention_lru_cluster2018.pth"
    )


if __name__ == "__main__":
    main()
