"""
检查 Alibaba cluster-trace batch_task / batch_instance CSV 是否与
process_cluster_trace_v2017.py 中的列定义一致。

用法:
  python inspect_cluster_trace_csv.py --data_dir "D:/trainData/cluster-trace-v2018/data_mini"
  python inspect_cluster_trace_csv.py --data_dir "D:/path/to/cluster-trace-v2017" --max_rows 1000
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# 与 process_cluster_trace_v2017.py 完全一致
V2017_BATCH_TASK = [
    "create_ts",
    "end_ts",
    "job_id",
    "task_id",
    "instance_num",
    "status",
    "plan_cpu",
    "plan_mem",
]
V2017_BATCH_INSTANCE = [
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


def _first_line_n_cols(path: Path) -> tuple[str, int]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
    if not line:
        return "", 0
    # 与 pandas read_csv 行为接近：按逗号分列
    row = next(csv.reader([line.rstrip("\n\r")]))
    return line[:200] + ("..." if len(line) > 200 else ""), len(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument(
        "--max_rows",
        type=int,
        default=5,
        help="用 pandas 试读行数（仅小样本，避免大文件 OOM）",
    )
    args = ap.parse_args()
    d = Path(args.data_dir)
    bt = d / "batch_task.csv"
    bi = d / "batch_instance.csv"

    print("=" * 60)
    print("Cluster trace CSV 与 process_cluster_trace_v2017.py 兼容性检查")
    print("=" * 60)
    print(f"data_dir: {d.resolve()}\n")

    for name, path, v2017_cols in [
        ("batch_task", bt, V2017_BATCH_TASK),
        ("batch_instance", bi, V2017_BATCH_INSTANCE),
    ]:
        print("-" * 60)
        print(f"[{name}] 期望列数(2017 脚本): {len(v2017_cols)}")
        print(f"  期望列名: {v2017_cols}")
        if not path.is_file():
            print(f"  [缺失] 文件不存在: {path}")
            continue
        size = path.stat().st_size
        print(f"  文件大小: {size / 1024 / 1024:.2f} MB ({size:,} bytes)")
        snippet, ncols = _first_line_n_cols(path)
        print(f"  首行列数(逗号分隔): {ncols}")
        print(f"  首行预览: {snippet!r}")
        if ncols == len(v2017_cols):
            print("  [列数] 与 2017 脚本一致，仍建议核对列顺序与语义是否同为 create_ts 打头。")
        else:
            print(
                f"  [列数] 与 2017 脚本不一致 (得到 {ncols}，需要 {len(v2017_cols)})。"
                " 通常说明这是其它年份/变体 schema，不能直接用于 process_cluster_trace_v2017.py。"
            )
        # 小样本 pandas 试读：按 2017 无表头+固定列名
        try:
            import pandas as pd

            df = pd.read_csv(
                path,
                header=None,
                names=v2017_cols,
                nrows=args.max_rows,
                low_memory=False,
            )
            print(f"  试读 {args.max_rows} 行 (按 2017 列名强读):")
            print(f"    create_ts 非空比例: {df['create_ts'].notna().mean():.2f}")
            c0 = df["create_ts"].iloc[0] if len(df) else None
            print(f"    首行 create_ts 样本: {c0!r} (若为字符串/任务名则说明列错位)")
        except Exception as e:
            print(f"  试读失败: {e}")
        print()

    print("=" * 60)
    print("结论说明:")
    print("  - 训练脚本 process_cluster_trace_v2017.py 假设: 无表头，列顺序与 2017 官方")
    print("    batch_task(8 列) / batch_instance(12 列) 一致。")
    print("  - 若首行列数或首字段语义不符(例如以任务名字符串打头而 create_ts 应为时间戳),")
    print("    需编写 cluster-trace-v2018 专用解析或列映射，再生成 JSONL。")
    print("=" * 60)


if __name__ == "__main__":
    main()
