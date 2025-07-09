#!/bin/bash
# GTN (Graph Transformer Networks) for WebQSP dataset

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0

# 训练参数
ENTITY_DIM=50  # 实体维度
NUM_EPOCH=200  # 训练轮数
BATCH_SIZE=8   # 批次大小
EVAL_EVERY=2   # 评估间隔

# GTN特有参数
NUM_ITER=3     # 迭代次数
NUM_INS=2      # 指令数量
NUM_GNN=3      # GNN层数
NUM_HEADS=8    # 注意力头数量
NUM_GTN_LAYERS=2  # GTN层数
D_FF=200       # 前馈网络隐藏层维度
GTN_DROPOUT=0.1  # Dropout比例

# 数据和模型相关参数
DATA_FOLDER=data/webqsp/
LM=sbert  # 语言模型类型
MODEL=GTN  # 模型名称

# 训练命令
echo "开始训练GTN模型..."
python main.py $MODEL \
    --entity_dim $ENTITY_DIM \
    --num_epoch $NUM_EPOCH \
    --batch_size $BATCH_SIZE \
    --eval_every $EVAL_EVERY \
    --data_folder $DATA_FOLDER \
    --lm $LM \
    --num_iter $NUM_ITER \
    --num_ins $NUM_INS \
    --num_gnn $NUM_GNN \
    --num_heads $NUM_HEADS \
    --num_gtn_layers $NUM_GTN_LAYERS \
    --d_ff $D_FF \
    --gtn_dropout $GTN_DROPOUT \
    --relation_word_emb True \
    --name webqsp_gtn

# 评估命令
echo "开始评估GTN模型..."
python main.py $MODEL \
    --entity_dim $ENTITY_DIM \
    --num_epoch $NUM_EPOCH \
    --batch_size $BATCH_SIZE \
    --eval_every $EVAL_EVERY \
    --data_folder $DATA_FOLDER \
    --lm $LM \
    --num_iter $NUM_ITER \
    --num_ins $NUM_INS \
    --num_gnn $NUM_GNN \
    --num_heads $NUM_HEADS \
    --num_gtn_layers $NUM_GTN_LAYERS \
    --d_ff $D_FF \
    --gtn_dropout $GTN_DROPOUT \
    --relation_word_emb True \
    --load_experiment GTN_webqsp.ckpt \
    --is_eval \
    --name webqsp_gtn_eval 