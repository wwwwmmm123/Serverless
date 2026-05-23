"""
Attention-LRU 模型推理服务

提供 HTTP API 供 Rust 调用

使用方式:
    python serve_model.py --model models/attention_lru.pth --port 5000
"""

from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F_nn
import argparse
import sys
import os
import time
from typing import Optional

sys.path.append(os.path.dirname(__file__))
from attention_lru import AttentionLRU, extract_features

app = Flask(__name__)

# 全局模型
model = None
model_load_time = None


def _align_feature_width(x: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Rust 端仍为 8 维时，右侧补零以匹配 12 维 DAG 模型。"""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    c = x.shape[-1]
    if c < feature_dim:
        x = F_nn.pad(x, (0, feature_dim - c))
    elif c > feature_dim:
        x = x[..., :feature_dim]
    return x


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_load_time': model_load_time
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    预测驱逐决策
    
    Request JSON:
    {
        "features": [[feat1, feat2, ...], [feat1, feat2, ...], ...],
        "can_be_evicted": [true, false, true, ...]
    }
    
    Response JSON:
    {
        "evict_index": 2,
        "scores": [0.8, 0.3, 0.6, ...],
        "inference_time_ms": 2.5
    }
    """
    global model
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        start_time = time.time()
        
        # 解析请求
        data = request.get_json()
        features = torch.tensor(data['features'], dtype=torch.float32)
        features = _align_feature_width(features, model.feature_dim)
        can_be_evicted = torch.tensor(data.get('can_be_evicted', [True] * len(features)), dtype=torch.bool)
        
        # 推理
        evict_idx = model.select_eviction(features, can_be_evicted)
        
        # 获取所有分数（用于调试）
        with torch.no_grad():
            scores = model(features)
            scores = scores.tolist() if scores.dim() == 1 else scores.squeeze(0).tolist()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return jsonify({
            'evict_index': evict_idx,
            'scores': scores,
            'inference_time_ms': round(inference_time, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/attention', methods=['POST'])
def get_attention():
    """
    获取注意力权重（用于可视化）
    
    Request JSON:
    {
        "features": [[feat1, feat2, ...], ...]
    }
    
    Response JSON:
    {
        "attention_weights": [[0.5, 0.1, ...], [0.2, 0.6, ...], ...]
    }
    """
    global model
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        features = torch.tensor(data['features'], dtype=torch.float32)
        features = _align_feature_width(features, model.feature_dim)
        
        # 获取注意力权重
        attn_weights = model.get_attention_weights(features)
        
        return jsonify({
            'attention_weights': attn_weights.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def load_model(model_path: str, feature_dim_override: Optional[int] = None):
    """加载模型"""
    global model, model_load_time
    
    print(f"Loading model from {model_path}...")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        fdim = feature_dim_override if feature_dim_override is not None else int(
            checkpoint.get("feature_dim", 8)
        )
        model = AttentionLRU(
            feature_dim=fdim,
            hidden_dim=64,
            num_heads=4
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        model_load_time = time.strftime('%Y-%m-%d %H:%M:%S')
        
        total_params = sum(p.numel() for p in model.parameters())
        print("[OK] Model loaded successfully")
        print(f"  - feature_dim: {fdim}")
        print(f"  - Parameters: {total_params:,}")
        print(f"  - Model size: ~{total_params * 4 / 1024:.1f} KB")
        
        # 训练统计
        if 'train_losses' in checkpoint:
            print(f"  - Training steps: {len(checkpoint['train_losses'])}")
            if len(checkpoint['train_losses']) > 0:
                print(f"  - Final loss: {checkpoint['train_losses'][-1]:.4f}")
                print(f"  - Final accuracy: {checkpoint['train_accuracies'][-1]:.3f}")
        
        return True
    
    except Exception as e:
        print(f"[ERR] Failed to load model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Attention-LRU Model Serving')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to serve on (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument(
        '--feature_dim',
        type=int,
        default=None,
        help='Override feature_dim in checkpoint (older models may omit this key)',
    )
    
    args = parser.parse_args()
    
    print(f"""
{'='*60}
Attention-LRU Model Server
{'='*60}
Configuration:
  - Model: {args.model}
  - Host: {args.host}
  - Port: {args.port}
  - feature_dim override: {args.feature_dim}
{'='*60}
""")
    
    # 加载模型
    if not load_model(args.model, args.feature_dim):
        print("\nPlease train a model first:")
        print("  python train_online.py --episodes 100")
        return
    
    # 启动服务
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print(f"  - Health check: http://{args.host}:{args.port}/health")
    print(f"  - Prediction API: http://{args.host}:{args.port}/predict")
    print(f"  - Attention API: http://{args.host}:{args.port}/attention")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
