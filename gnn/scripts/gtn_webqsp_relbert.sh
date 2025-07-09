#!/bin/bash

###GTN+SBERT training
python main.py GTN --entity_dim 50 --num_epoch 10 --batch_size 4 --eval_every 2  \
--lm relbert --num_iter 2 --num_ins 3 --num_gnn 3 --num_heads 2 --num_gtn_layers 2 \
--d_ff 200 --gtn_dropout 0.1 --name webqsp --data_eff \
--experiment_name webqsp-gtn-relbert-debug --data_folder data/webqsp/ --warmup_epoch 8
# --is_eval --load_experiment webqsp-gtn-relbert-final.ckpt 

###GTN+LMSR training
# python main.py GTN --entity_dim 50 --num_epoch 10 --batch_size 4 --eval_every 2  \
# --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3 --num_heads 2 --num_gtn_layers 2 \
# --d_ff 200 --gtn_dropout 0.1 --name webqsp --data_eff \
# --experiment_name webqsp-gtn-lmsr --data_folder data/webqsp/ --num_epoch 10 #--warmup_epoch 8

###Evaluate webqsp - 注释此部分，直到训练完成后生成检查点文件
# python main.py GTN --entity_dim 50 --num_epoch 10 --batch_size 4 --eval_every 2 \
# --data_folder data/webqsp/ --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3 \
# --num_heads 2 --num_gtn_layers 2 --d_ff 200 --gtn_dropout 0.1 --data_eff \
# --relation_word_emb True --load_experiment webqsp-gtn-relbert-debug-h1.ckpt --is_eval \
# --name webqsp --experiment_name webqsp-gtn-relbert-eval 