# 阿里巴巴微服务 Trace（DAG）离线训练指南

本指南说明如何从 **Alibaba cluster-trace-microservices**（含 MS_CallGraph）生成 JSONL 数据集，并训练 **Attention-LRU**（或 SimpleMLP）。所有命令默认在 **`ml_training`** 目录下执行。

## 前置条件

1. 已下载并解压 **cluster-trace-microservices-v2021**（或等价目录），其中包含可读的 **CallGraph** 表（脚本会在 `--data_dir` 下递归查找 `*CallGraph*` 等文件）。
2. Python 环境已安装项目依赖（含 `torch`、`pandas`、`tqdm` 等）。
3. 若 CallGraph 以 `.tar.gz` 形式存在，需先解压到 `--data_dir` 可访问的路径，否则 `process_alibaba_trace.py` 可能找不到表文件。

## 一步：生成训练用 JSONL

脚本：`process_alibaba_trace.py`  
默认输出：`datasets/alibaba_dag_cache_decisions.jsonl`（12 维特征，含 DAG 相关字段，与 `train_offline.py` 中 `feature_dim=12` 推断一致）。

**PowerShell（请把 `DATA_DIR` 换成你的 Trace 根目录）：**

```powershell
cd d:\serverless\serverless_sim\ml_training

python process_alibaba_trace.py `
  --data_dir "D:\path\to\cluster-trace-microservices-v2021" `
  --output datasets/alibaba_dag_cache_decisions.jsonl `
  --num_samples 50000 `
  --max_callgraph_rows 2000000 `
  --cache_capacity 10 `
  --seed 42
```

常用参数说明：

| 参数 | 含义 |
|------|------|
| `--data_dir` | Trace 根目录（内含或可递归找到 CallGraph 数据文件） |
| `--output` | 输出 JSONL 路径（相对 `ml_training` 或绝对路径） |
| `--num_samples` | 生成的监督样本条数 |
| `--max_callgraph_rows` | 从 CallGraph 最多读入行数（控内存/时间） |
| `--cache_capacity` | 每条样本的缓存槽位数（与训练时序列长度一致） |
| `--seed` | 随机种子 |

生成成功后终端会提示示例训练命令。

## 二步：训练 Attention-LRU（阿里巴巴数据）

`train_offline.py` 的 **`--dataset` 默认值** 已是 `datasets/alibaba_dag_cache_decisions.jsonl`，若你使用该默认输出路径，可省略 `--dataset`。

**推荐命令：**

```powershell
cd d:\serverless\serverless_sim\ml_training

python train_offline.py `
  --dataset datasets/alibaba_dag_cache_decisions.jsonl `
  --save_path models/attention_lru_alibaba_dag.pth `
  --epochs 50 `
  --batch_size 64 `
  --device cuda
```

说明：

- 训练曲线默认保存为 **`models/attention_lru_alibaba_dag_curves.png`**（与 `--save_path` 同名加 `_curves`）。也可用 `--plot_path` 自定义。
- 摘要与曲线中的 **Train** 指标在 **`model.eval()`** 下对训练集再算一遍，与 **Val** 可比；详见 `train_offline.py` 文件头注释。
- GPU 可用时将 `--device cpu` 改为 `--device cuda` 或删除该参数（脚本会按环境自动选择）。
- **`--seed`**：控制训练/验证划分与 DataLoader 打乱，默认 42。

## 可选：SimpleMLP 对照训练

```powershell
cd d:\serverless\serverless_sim\ml_training

python train_simple_mlp.py `
  --dataset datasets/alibaba_dag_cache_decisions.jsonl `
  --epochs 50 `
  --batch_size 64 `
  --save_path models/simple_mlp_alibaba_dag.pth `
  --plot_path models/simple_mlp_alibaba_dag_curves.png
```

## 批处理一键流程（Windows）

仓库内已提供 **`train_alibaba_dag.bat`**：依次执行「生成 JSONL → SimpleMLP → Attention-LRU」。使用前请用文本编辑器修改其中的 **`DATA_DIR`** 为你的本机 Trace 路径。

## 与 Cluster 2017 流程的区别

| 项目 | 阿里巴巴 DAG | Cluster 2017 |
|------|----------------|--------------|
| 数据脚本 | `process_alibaba_trace.py` | `process_cluster_trace_v2017.py` |
| 典型输出 | `datasets/alibaba_dag_cache_decisions.jsonl` | 如 `datasets/cluster2017_batch_cache_decisions.jsonl` |
| 特征维度 | JSONL 含 DAG 字段 → 推断 **12** | 通常 **8** |

训练命令形式相同，仅 **`--dataset`** 与 **`--save_path`** / **`--plot_path`** 换成对应文件名即可。

## 推理服务（可选）

训练完成后可使用项目中的推理脚本（具体以仓库内 `serve_model.py` 说明为准），例如：

```powershell
python serve_model.py --model models/attention_lru_alibaba_dag.pth --port 5000
```

（若端口或参数与当前代码不一致，以 `serve_model.py` 的 `--help` 为准。）
