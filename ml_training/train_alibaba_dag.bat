@echo off
REM Alibaba 微服务 DAG Trace -> JSONL -> Attention-LRU / SimpleMLP
REM 数据目录可改；默认与 ALIBABA_DAG_INTEGRATION_PLAN.md 一致

cd /d "%~dp0"

set DATA_DIR=C:\Users\王盟啊\Downloads\clusterdata-master\clusterdata-master\cluster-trace-microservices-v2021
set OUT_JSONL=datasets\alibaba_dag_cache_decisions.jsonl

echo ======================================================================
echo [1/3] 生成训练数据 (process_alibaba_trace.py)
echo ======================================================================
python process_alibaba_trace.py --data_dir "%DATA_DIR%" --output "%OUT_JSONL%" --num_samples 50000 --max_callgraph_rows 2000000
if errorlevel 1 (
    echo [ERR] 数据处理失败。请确认已解压 MSCallGraph 下的数据，或修改本 bat 中的 DATA_DIR。
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo [2/3] SimpleMLP 训练 (推荐先跑通)
echo ======================================================================
python train_simple_mlp.py --dataset "%OUT_JSONL%" --epochs 50 --batch_size 64 --save_path models\simple_mlp_alibaba_dag.pth --plot_path models\simple_mlp_alibaba_dag_curves.png --summary_path models\simple_mlp_alibaba_dag_last_run.json
if errorlevel 1 (
    echo [ERR] SimpleMLP 训练失败
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo [3/3] Attention-LRU 离线训练
echo ======================================================================
python train_offline.py --dataset "%OUT_JSONL%" --epochs 50 --batch_size 64 --save_path models\attention_lru_alibaba_dag.pth
if errorlevel 1 (
    echo [ERR] Attention-LRU 训练失败
    pause
    exit /b 1
)

echo.
echo [OK] 完成。推理服务示例:
echo   python serve_model.py --model models\attention_lru_alibaba_dag.pth --port 5000
pause
