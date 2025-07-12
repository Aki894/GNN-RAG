#!/usr/bin/env python3
"""
优化的训练脚本，包含GPU监控、内存管理和性能优化
"""

import argparse
import os
import time
import psutil
import GPUtil
from utils import create_logger
import torch
import numpy as np
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Gradformer_main'))

from train_model import Trainer_KBQA
from parsing import add_parse_args

def monitor_gpu():
    """监控GPU使用情况"""
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.memoryUtil*100:.1f}% memory, {gpu.load*100:.1f}% load")
        return gpus[0] if gpus else None
    except:
        return None

def monitor_memory():
    """监控系统内存使用情况"""
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.percent:.1f}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")

def optimize_torch_settings():
    """优化PyTorch设置以提高性能"""
    # 启用TF32以提高性能（仅适用于Ampere架构）
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 设置cudnn基准测试
    torch.backends.cudnn.benchmark = True
    
    # 设置确定性计算（可选，会降低性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_optimal_batch_size(gpu_memory_gb):
    """根据GPU内存确定最优批处理大小"""
    if gpu_memory_gb >= 24:
        return 64
    elif gpu_memory_gb >= 16:
        return 48
    elif gpu_memory_gb >= 12:
        return 32
    elif gpu_memory_gb >= 8:
        return 24
    else:
        return 16

def main():
    parser = argparse.ArgumentParser(description='Optimized GNN-RAG Training')
    add_parse_args(parser)
    
    # 添加优化相关参数
    parser.add_argument('--monitor_gpu', action='store_true', help='Enable GPU monitoring')
    parser.add_argument('--auto_batch_size', action='store_true', help='Auto-adjust batch size based on GPU memory')
    parser.add_argument('--profile_memory', action='store_true', help='Enable memory profiling')
    
    args = parser.parse_args()
    
    # 优化PyTorch设置
    optimize_torch_settings()
    
    # 检查CUDA可用性
    args.use_cuda = torch.cuda.is_available()
    if args.use_cuda:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f}GB")
        
        # 自动调整批处理大小
        if args.auto_batch_size:
            optimal_batch_size = get_optimal_batch_size(gpu_memory)
            if args.batch_size != optimal_batch_size:
                print(f"Adjusting batch size from {args.batch_size} to {optimal_batch_size}")
                args.batch_size = optimal_batch_size
    else:
        print("Using CPU")
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置实验名称
    if args.experiment_name == None:
        timestamp = str(int(time.time()))
        args.experiment_name = "{}-{}-{}".format(
            args.dataset,
            args.model_name,
            timestamp,
        )
    
    # 创建检查点目录
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    # 创建日志记录器
    logger = create_logger(args)
    
    # 记录系统信息
    logger.info(f"System: {os.uname().sysname} {os.uname().release}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    if args.use_cuda:
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {gpu_memory:.1f}GB")
    
    # 监控初始状态
    if args.monitor_gpu:
        print("\n=== Initial GPU Status ===")
        monitor_gpu()
        monitor_memory()
        print("========================\n")
    
    # 创建训练器
    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger=logger)
    
    # 训练或评估
    if not args.is_eval:
        print("Starting optimized training...")
        start_time = time.time()
        
        try:
            trainer.train(0, args.num_epoch - 1)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining error: {e}")
            raise
        finally:
            end_time = time.time()
            training_time = end_time - start_time
            logger.info(f"Total training time: {training_time/3600:.2f} hours")
            
            if args.monitor_gpu:
                print("\n=== Final GPU Status ===")
                monitor_gpu()
                monitor_memory()
                print("========================\n")
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        trainer.evaluate_single(ckpt_path)

if __name__ == '__main__':
    main() 