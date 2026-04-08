@echo off
REM Azure Functions 数据集训练一键启动脚本
REM 作者: AI Assistant
REM 日期: 2026-03-26

echo ======================================================================
echo Azure Functions Attention-LRU 训练脚本
echo ======================================================================
echo.

cd /d "%~dp0"

echo 当前目录: %cd%
echo.

REM 步骤 1: 检查 Python
echo [步骤 1/4] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)
python --version
echo.

REM 步骤 2: 安装依赖
echo [步骤 2/4] 安装 Python 依赖...
echo 这可能需要几分钟...
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)
echo.
echo [完成] 依赖安装成功
echo.

REM 步骤 3: 处理 Azure 数据集
echo [步骤 3/4] 处理 Azure Functions 数据集...
echo 这将需要 5-10 分钟，请耐心等待...
echo.
python process_azure_real_data.py
if errorlevel 1 (
    echo [错误] 数据处理失败
    echo 请检查数据集是否在正确位置: C:\Users\王盟啊\Downloads\AzurePublicDataset-master\data
    pause
    exit /b 1
)
echo.
echo [完成] 数据处理成功
echo.

REM 步骤 4: 训练模型
echo [步骤 4/4] 训练 Attention-LRU 模型...
echo 这将需要 3-5 分钟...
echo.
python train_simple_mlp.py --dataset datasets/azure_real_cache_decisions.jsonl --epochs 50 --batch_size 64
if errorlevel 1 (
    echo [错误] 模型训练失败
    pause
    exit /b 1
)
echo.

REM 完成
echo ======================================================================
echo [成功] 所有步骤完成！
echo ======================================================================
echo.
echo 生成的文件:
echo   - 训练数据: datasets\azure_real_cache_decisions.jsonl
echo   - 模型文件: models\attention_lru_offline.pth
echo   - 训练曲线: models\offline_training_curves.png
echo.
echo 查看训练曲线:
start models\offline_training_curves.png
echo.
echo 按任意键退出...
pause >nul
