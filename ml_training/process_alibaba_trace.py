"""
从阿里巴巴微服务 Trace（含 MS_CallGraph）生成离线训练用 JSONL。

默认数据根目录与集成文档一致；可通过 --data_dir 覆盖。
不依赖解压后固定子路径名：会在目录下递归查找疑似 CallGraph 的 CSV/无扩展名表文件。

用法:
  python process_alibaba_trace.py ^
    --data_dir "C:\\Users\\王盟啊\\Downloads\\clusterdata-master\\clusterdata-master\\cluster-trace-microservices-v2021" ^
    --output datasets/alibaba_dag_cache_decisions.jsonl ^
    --num_samples 50000 ^
    --max_callgraph_rows 2000000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _find_callgraph_files(data_dir: Path) -> List[Path]:
    """在 data_dir 下查找 CallGraph 表文件。"""
    candidates: List[Path] = []
    patterns = (
        "*CallGraph*",
        "*callgraph*",
        "*MSCallGraph*",
    )
    for pat in patterns:
        candidates.extend(data_dir.rglob(pat))
    # 常见：解压后的 .csv 或无名数据文件
    for sub in ("MSCallGraph", "MSCallGraph_Table", "callgraph", "CallGraph"):
        p = data_dir / sub
        if p.is_dir():
            for ext in ("*.csv", "*.txt", "*.tsv"):
                candidates.extend(p.rglob(ext))
    # 去重、只保留文件
    seen: Set[Path] = set()
    out: List[Path] = []
    for c in candidates:
        if c.is_file() and c.suffix.lower() in (".csv", ".txt", ".tsv", ""):
            rp = c.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(c)
    return sorted(out, key=lambda x: str(x))


def _read_callgraph_table(path: Path, max_rows: int) -> pd.DataFrame:
    """读取 CallGraph 表；自动尝试分隔符与列名。"""
    read_kw: Dict[str, Any] = {"nrows": max_rows, "low_memory": False}
    for sep in (",", "\t", "|"):
        try:
            df = pd.read_csv(path, sep=sep, **read_kw)
            if len(df.columns) >= 5:
                return df
        except Exception:
            continue
    return pd.read_csv(path, nrows=max_rows, low_memory=False)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    colmap = {
        "trace_id": "traceid",
        "trace id": "traceid",
        "rpc_id": "rpcid",
        "um": "um",
        "dm": "dm",
        "rt": "rt",
    }
    for a, b in colmap.items():
        if a in df.columns and b not in df.columns:
            df.rename(columns={a: b}, inplace=True)
    need = {"traceid", "um", "dm"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CallGraph 表缺少列: {missing}，实际列: {list(df.columns)}")
    return df


def _invalid_ms(name: Any) -> bool:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return True
    s = str(name).strip().lower()
    if not s or s in ("nan", "none", "(?)", "?"):
        return True
    return False


def _build_trace_graphs(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    traceid -> {
        'nodes': set,
        'out': um -> set(dm),
        'in': dm -> set(um),
        'rt_sum': ms -> float,
        'rt_cnt': ms -> int,
    }
    """
    traces: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        tid = str(row["traceid"])
        um, dm = row["um"], row["dm"]
        if _invalid_ms(um) or _invalid_ms(dm):
            continue
        um_s, dm_s = str(um).strip(), str(dm).strip()
        try:
            rt = float(row["rt"]) if "rt" in row and pd.notna(row["rt"]) else 0.0
        except Exception:
            rt = 0.0
        if tid not in traces:
            traces[tid] = {
                "nodes": set(),
                "out": defaultdict(set),
                "inn": defaultdict(set),
                "rt_sum": defaultdict(float),
                "rt_cnt": defaultdict(int),
            }
        t = traces[tid]
        t["nodes"].add(um_s)
        t["nodes"].add(dm_s)
        t["out"][um_s].add(dm_s)
        t["inn"][dm_s].add(um_s)
        for ms in (um_s, dm_s):
            t["rt_sum"][ms] += abs(rt)
            t["rt_cnt"][ms] += 1
    return traces


def _bfs_depth(roots: List[str], out_adj: DefaultDict[str, Set[str]]) -> Dict[str, int]:
    depth: Dict[str, int] = {}
    q: deque = deque()
    for r in roots:
        depth[r] = 0
        q.append(r)
    while q:
        u = q.popleft()
        for v in out_adj.get(u, ()):
            nd = depth[u] + 1
            if v not in depth or nd < depth[v]:
                depth[v] = nd
                q.append(v)
    return depth


def _teacher_eviction_score(
    entry: Dict[str, Any],
    current_time: int,
    cold_w: float = 0.3,
    mem_w: float = 0.2,
    freq_w: float = 0.3,
    rec_w: float = 0.2,
) -> float:
    """与 process_azure_real_data / train 标签一致：分数越高越应被驱逐。"""
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


def _optional_resource_stats(data_dir: Path) -> Dict[str, Tuple[float, float]]:
    """若存在 MSResource 下 CSV，则聚合 msname -> (mean_cpu, mean_mem)。"""
    stats: Dict[str, Tuple[float, float]] = {}
    for sub in ("MSResource", "MS_Resource", "msresource"):
        base = data_dir / sub
        if not base.is_dir():
            continue
        for csv_path in list(base.rglob("*.csv"))[:20]:
            try:
                chunk = pd.read_csv(csv_path, nrows=500_000, low_memory=False)
            except Exception:
                continue
            cols = {c.lower(): c for c in chunk.columns}
            name_col = cols.get("msname") or cols.get("ms_name")
            cpu_col = cols.get("cpu_utilization") or cols.get("cpu")
            mem_col = cols.get("memory_utilization") or cols.get("memory")
            if not name_col or not cpu_col or not mem_col:
                continue
            g = chunk.groupby(name_col)[[cpu_col, mem_col]].mean()
            for ms, row in g.iterrows():
                if _invalid_ms(ms):
                    continue
                stats[str(ms).strip()] = (float(row[cpu_col]), float(row[mem_col]))
    return stats


class AlibabaDagGenerator:
    def __init__(
        self,
        traces: Dict[str, Dict[str, Any]],
        resource_stats: Dict[str, Tuple[float, float]],
        cache_capacity: int = 10,
        seed: int = 42,
    ):
        self.traces = traces
        self.resource_stats = resource_stats
        self.cache_capacity = cache_capacity
        random.seed(seed)
        np.random.seed(seed)

        self.valid_trace_ids = [
            tid
            for tid, t in traces.items()
            if len(t["nodes"]) >= cache_capacity + 1
        ]
        if not self.valid_trace_ids:
            raise ValueError(
                "没有足够大的 trace（节点数 >= cache_capacity+1）。"
                "请增大 --max_callgraph_rows 或检查数据是否已解压。"
            )

        # 全局调用频次（用于 call_frequency 特征）
        self.global_call_count: DefaultDict[str, int] = defaultdict(int)
        for t in traces.values():
            for ms in t["nodes"]:
                self.global_call_count[ms] += 1
        max_c = max(self.global_call_count.values()) if self.global_call_count else 1

        self._max_call_count = float(max_c)

    def _make_entry(
        self,
        ms: str,
        trace: Dict[str, Any],
        depth: Dict[str, int],
        roots: Set[str],
        current_time: int,
    ) -> Dict[str, Any]:
        out_adj = trace["out"]
        in_adj = trace["inn"]
        d_out = len(out_adj.get(ms, ()))
        d_in = len(in_adj.get(ms, ()))
        dep = depth.get(ms, 0)
        is_root = ms in roots

        rt_sum = trace["rt_sum"].get(ms, 0.0)
        rt_cnt = max(1, trace["rt_cnt"].get(ms, 1))
        avg_rt = rt_sum / rt_cnt

        if ms in self.resource_stats:
            cpu_raw, mem_raw = self.resource_stats[ms]
        else:
            cpu_raw = float(np.clip(np.log1p(avg_rt) / 10.0, 0.0, 1.0))
            mem_raw = float(np.clip(avg_rt / 500.0, 0.0, 1.0))

        cpu = float(np.clip(cpu_raw, 0.0, 1.0))
        memory = float(np.clip(mem_raw, 0.0, 1.0))
        cold_start_time = float(np.clip(avg_rt / 200.0, 0.0, 1.0))
        output_size = float(np.clip(memory * 0.3, 0.0, 1.0))
        call_frequency = float(self.global_call_count.get(ms, 1) / self._max_call_count)

        last_access_time = max(
            0, current_time - int(np.random.randint(0, min(current_time + 1, 500)))
        )
        access_count = min(int(self.global_call_count.get(ms, 1)), 100)

        return {
            "ms_name": ms[:16] + "..." if len(ms) > 16 else ms,
            "cpu": cpu,
            "memory": memory,
            "cold_start_time": cold_start_time,
            "output_size": output_size,
            "call_frequency": call_frequency,
            "last_access_time": last_access_time,
            "access_count": access_count,
            "is_dag_root": is_root,
            "downstream_count": d_out,
            "upstream_count": d_in,
            "dag_depth": dep,
        }

    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        decisions: List[Dict[str, Any]] = []
        for i in tqdm(range(num_samples), desc="Alibaba DAG samples"):
            tid = random.choice(self.valid_trace_ids)
            tr = self.traces[tid]
            nodes = list(tr["nodes"])
            if len(nodes) < self.cache_capacity + 1:
                continue

            cache_nodes = random.sample(nodes, self.cache_capacity)
            pool = [n for n in nodes if n not in cache_nodes]
            if not pool:
                continue

            out_full: DefaultDict[str, Set[str]] = defaultdict(set)
            in_full: DefaultDict[str, Set[str]] = defaultdict(set)
            for u, vs in tr["out"].items():
                for v in vs:
                    out_full[u].add(v)
                    in_full[v].add(u)
            roots = {n for n in tr["nodes"] if len(in_full[n]) == 0}
            if not roots:
                roots = {min(tr["nodes"], key=lambda x: (len(in_full[x]), x))}

            depth = _bfs_depth(list(roots), out_full)

            current_time = i * 10 + 1000
            cache_entries = []
            for idx, ms in enumerate(cache_nodes):
                e = self._make_entry(ms, tr, depth, roots, current_time)
                e["fn_id"] = idx
                cache_entries.append(e)

            scores = [_teacher_eviction_score(e, current_time) for e in cache_entries]
            optimal_evict_idx = int(np.argmax(scores))

            decisions.append(
                {
                    "decision_id": i,
                    "dag_source": "alibaba",
                    "traceid_sample": tid[:12],
                    "timestamp": current_time,
                    "cache_entries": cache_entries,
                    "optimal_evict_idx": optimal_evict_idx,
                    "can_be_evicted": [True] * self.cache_capacity,
                }
            )
        return decisions


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(rows):,} lines -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Alibaba MS trace -> cache JSONL")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\Users\王盟啊\Downloads\clusterdata-master\clusterdata-master\cluster-trace-microservices-v2021",
        help="cluster-trace-microservices-v2021 根目录（含 MSCallGraph 等）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/alibaba_dag_cache_decisions.jsonl",
        help="输出 JSONL 路径（相对 ml_training 或绝对路径）",
    )
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--cache_capacity", type=int, default=10)
    parser.add_argument(
        "--max_callgraph_rows",
        type=int,
        default=2_000_000,
        help="从 CallGraph 最多读取行数（控制内存与时间）",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir 不存在: {data_dir}")

    files = _find_callgraph_files(data_dir)
    if not files:
        print(
            "[WARN] 未找到 CallGraph 数据文件。请先在该目录下解压 MSCallGraph 下的 .tar.gz，"
            "或指定正确 --data_dir。"
        )
        print(f"  搜索根目录: {data_dir.resolve()}")
        raise SystemExit(1)

    print(f"[OK] 将读取最多 {args.max_callgraph_rows:,} 行，候选文件数: {len(files)}")
    frames: List[pd.DataFrame] = []
    remaining = args.max_callgraph_rows
    for fp in files:
        if remaining <= 0:
            break
        try:
            df = _read_callgraph_table(fp, min(remaining, args.max_callgraph_rows))
            df = _normalize_columns(df)
            frames.append(df)
            remaining -= len(df)
            print(f"  + {fp.name}: {len(df):,} 行")
        except Exception as e:
            print(f"  [SKIP] {fp}: {e}")

    if not frames:
        raise SystemExit("未能读取任何 CallGraph 数据。")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"[OK] 合计行数: {len(df_all):,}，构建 trace 图...")
    traces = _build_trace_graphs(df_all)
    print(f"[OK] trace 数: {len(traces):,}")

    res_stats = _optional_resource_stats(data_dir)
    if res_stats:
        print(f"[OK] 资源表聚合到 {len(res_stats):,} 个微服务")
    else:
        print("[INFO] 未找到 MSResource CSV，将用 RT 推导 cpu/memory 近似特征")

    gen = AlibabaDagGenerator(
        traces, res_stats, cache_capacity=args.cache_capacity, seed=args.seed
    )
    decisions = gen.generate(args.num_samples)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    save_jsonl(decisions, out_path)
    print("[OK] 完成。训练示例:")
    print(
        f"  python train_simple_mlp.py --dataset {out_path.name} "
        f"--save_path models/simple_mlp_alibaba_dag.pth"
    )
    print(
        f"  python train_offline.py --dataset {out_path.name} "
        f"--save_path models/attention_lru_alibaba_dag.pth"
    )


if __name__ == "__main__":
    main()
