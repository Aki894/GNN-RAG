#!/bin/bash

# 优化的GNN-RAG训练脚本
# 包含GPU监控、自动批处理大小调整和性能优化

echo "=== GNN-RAG Optimized Training Script ==="
echo "Starting optimized training with performance enhancements..."

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "No NVIDIA GPU detected, using CPU"
fi

# 设置环境变量以优化性能
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# 安装必要的监控包（如果未安装）
pip install psutil GPUtil --quiet

# 运行优化的训练脚本
python train_optimized.py ReaRev \
    --name webqsp \
    --data_folder data/webqsp/ \
    --batch_size 32 \
    --test_batch_size 32 \
    --num_epoch 50 \
    --eval_every 2 \
    --lr 0.0005 \
    --gradient_clip 1.0 \
    --use_amp \
    --monitor_gpu \
    --auto_batch_size \
    --num_workers 4 \
    --seed 42 \
    --experiment_name "optimized_rearev_$(date +%Y%m%d_%H%M%S)" \
    --checkpoint_dir checkpoint/optimized/ \
    --log_level info

echo "=== Training completed ===" 