"""
Alibaba Cluster Trace v2017 -> 缓存决策 JSONL（12 维含启发式 DAG 特征）

重要说明（与 microservices-2021 不同）：
  官方 schema 中 batch_task / batch_instance **不包含** RPC 调用边。
  本脚本在同一 job_id 内按 task 创建时间排序，**假设任务沿一条链依次依赖**（启发式），
  仅用于在较小数据集上得到与 train_offline / train_simple_mlp 兼容的 12 维特征。
  论文中应明确写为 "job-level heuristic chain" 而非真实调用图。

默认数据目录示例：
  .../clusterdata-master/cluster-trace-v2017

需要文件（无表头 CSV，与官方一致）：
  - batch_task.csv
  - batch_instance.csv（用于时长/频次近似）

用法:
  python process_cluster_trace_v2017.py ^
    --data_dir "D:\\data\\cluster-trace-v2017" ^
    --output datasets/cluster2017_batch_cache_decisions.jsonl ^
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
    cold_w: float = 0.3,
    mem_w: float = 0.2,
    freq_w: float = 0.3,
    rec_w: float = 0.2,
) -> float:
    recency = (current_time - entry["last_access_time"]) / max(current_time, 1)
    frequency = entry["call_frequency"]
    cold_start = entry["cold_start_time"]
    memory = entry["memory"]
    return (
        cold_w * cold_start
        + mem_w * memory
        + freq_w * (1.0 / (frequency + 1e-6))
        + rec_w * recency
    )


def _load_batch_task(path: Path, max_rows: int) -> pd.DataFrame:
    cols = [
        "create_ts",
        "end_ts",
        "job_id",
        "task_id",
        "instance_num",
        "status",
        "plan_cpu",
        "plan_mem",
    ]
    df = pd.read_csv(
        path,
        header=None,
        names=cols,
        nrows=max_rows,
        low_memory=False,
    )
    for c in ("create_ts", "end_ts", "job_id", "task_id", "instance_num", "plan_cpu", "plan_mem"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["plan_cpu"] = df["plan_cpu"].fillna(0.0).clip(lower=0.0)
    df["plan_mem"] = df["plan_mem"].fillna(0.0).clip(lower=0.0, upper=1.0)
    df["instance_num"] = df["instance_num"].fillna(0).astype(np.int64)
    df = df.dropna(subset=["job_id", "task_id", "create_ts"])
    return df


def _load_batch_instance(path: Path, max_rows: int) -> pd.DataFrame:
    cols = [
        "start_ts",
        "end_ts",
        "job_id",
        "task_id",
        "machine_id",
        "status",
        "seq_no",
        "total_seq_no",
        "real_cpu_max",
        "real_cpu_avg",
        "real_mem_max",
        "real_mem_avg",
    ]
    df = pd.read_csv(
        path,
        header=None,
        names=cols,
        nrows=max_rows,
        low_memory=False,
    )
    for c in (
        "start_ts",
        "end_ts",
        "job_id",
        "task_id",
        "machine_id",
        "seq_no",
        "total_seq_no",
        "real_cpu_max",
        "real_cpu_avg",
        "real_mem_max",
        "real_mem_avg",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["job_id", "task_id"])
    return df


def _duration_stats(inst: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    """(job_id, task_id) -> 平均运行时长（秒），用于冷启动近似。"""
    df = inst.copy()
    df = df[(df["start_ts"] > 0) & (df["end_ts"] > df["start_ts"])]
    if df.empty:
        return {}
    df["dur"] = df["end_ts"] - df["start_ts"]
    g = df.groupby(["job_id", "task_id"])["dur"].mean()
    return {tuple(k): float(v) for k, v in g.items()}


def _task_frequency(inst: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    return (
        inst.groupby(["job_id", "task_id"])
        .size()
        .to_dict()
    )


def build_jobs_from_tasks(task_df: pd.DataFrame) -> Dict[int, List[Dict[str, Any]]]:
    """job_id -> 按 create_ts 排序的任务列表（含特征字段）。"""
    jobs: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for _, row in task_df.iterrows():
        jid = int(row["job_id"])
        tid = int(row["task_id"])
        jobs[jid].append(
            {
                "job_id": jid,
                "task_id": tid,
                "create_ts": int(row["create_ts"]),
                "plan_cpu": float(row["plan_cpu"]),
                "plan_mem": float(row["plan_mem"]),
                "instance_num": int(row["instance_num"]),
            }
        )
    for jid in jobs:
        jobs[jid].sort(key=lambda x: (x["create_ts"], x["task_id"]))
    return dict(jobs)


def generate_samples(
    jobs: Dict[int, List[Dict[str, Any]]],
    dur_map: Dict[Tuple[int, int], float],
    freq_map: Dict[Tuple[int, int], int],
    cache_capacity: int,
    num_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    valid_jobs = [jid for jid, tasks in jobs.items() if len(tasks) >= cache_capacity + 1]
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
    for i in tqdm(range(num_samples), desc="cluster-2017 samples"):
        jid = rng.choice(valid_jobs)
        tasks = jobs[jid]
        n = len(tasks)
        chosen_idx = sorted(rng.sample(range(n), cache_capacity))
        current_time = i * 10 + 10_000

        cache_entries: List[Dict[str, Any]] = []
        for slot, idx in enumerate(chosen_idx):
            t = tasks[idx]
            key = (t["job_id"], t["task_id"])
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
                    "job_id": jid,
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
                "dag_source": "alibaba_cluster_2017_batch_chain_heuristic",
                "job_id_sample": jid,
                "timestamp": current_time,
                "cache_entries": cache_entries,
                "optimal_evict_idx": optimal_evict_idx,
                "can_be_evicted": [True] * cache_capacity,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster trace v2017 -> cache JSONL")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="cluster-trace-v2017 目录（含 batch_task.csv, batch_instance.csv）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/cluster2017_batch_cache_decisions.jsonl",
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

    print(f"[OK] 读取 batch_task.csv (最多 {args.max_task_rows:,} 行)...")
    task_df = _load_batch_task(task_path, args.max_task_rows)
    print(f"[OK] 读取 batch_instance.csv (最多 {args.max_task_rows:,} 行)...")
    inst_df = _load_batch_instance(inst_path, args.max_task_rows)

    print("[OK] 构建 job -> tasks ...")
    jobs = build_jobs_from_tasks(task_df)
    print(f"[OK] job 数: {len(jobs):,}")

    dur_map = _duration_stats(inst_df)
    freq_map = _task_frequency(inst_df)
    print(f"[OK] 时长键: {len(dur_map):,}, 频次键: {len(freq_map):,}")

    decisions = generate_samples(
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
        f"--save_path models/simple_mlp_cluster2017.pth"
    )
    print(
        f"  python train_offline.py --dataset {out_path.name} "
        f"--save_path models/attention_lru_cluster2017.pth"
    )


if __name__ == "__main__":
    main()
