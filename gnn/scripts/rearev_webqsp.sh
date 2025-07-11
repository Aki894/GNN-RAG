
###ReaRev+SBERT training with GTN
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2  \
--lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name webqsp \
--experiment_name webqsp-rearev-relbert-res-psnr-gtn --data_folder data/webqsp/ --warmup_epoch 80 \
--use_gtn True --gtn_channels 2 --gtn_layers 1 --use_hybrid_reasoning True --use_residual True

###ReaRev+SBERT training without GTN (baseline)
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2  \
--lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name webqsp \
--experiment_name webqsp-rearev-relbert-res-psnr-baseline --data_folder data/webqsp/ --warmup_epoch 80 \
--use_gtn False

###ReaRev+LMSR training
# python main.py ReaRev  --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2  \
# --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name webqsp \
# --experiment_name prn_webqsp-rearev-lmsr  --data_folder data/webqsp/ --num_epoch 100 #--warmup_epoch 80


###Evaluate webqsp with GTN
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --data_folder data/webqsp/ \
--lm relbert --num_iter 2 --num_ins 3 --num_gnn 3 --relation_word_emb True \
--load_experiment webqsp-rearev-relbert-res-psnr-gtn-h1.ckpt --is_eval --name webqsp \
--experiment_name webqsp-rearev-relbert-res-psnr-gtn \
--use_gtn True --gtn_channels 2 --gtn_layers 1 --use_hybrid_reasoning True --use_residual True

###Evaluate webqsp without GTN (baseline)
python main.py ReaRev --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 --data_folder data/webqsp/ \
--lm relbert --num_iter 2 --num_ins 3 --num_gnn 3 --relation_word_emb True \
--load_experiment webqsp-rearev-relbert-res-psnr-baseline-h1.ckpt --is_eval --name webqsp \
--experiment_name webqsp-rearev-relbert-res-psnr-baseline \
--use_gtn False