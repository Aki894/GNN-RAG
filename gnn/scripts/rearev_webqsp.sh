
###ReaRev+SBERT training
python main.py ReaRev --entity_dim 64 --num_epoch 100 --batch_size 8 --eval_every 2  \
--lm sbert --num_iter 2 --num_ins 3 --num_gnn 3  --name webqsp \
--experiment_name webqsp-rearev-relbert-res-psnr --data_folder data/webqsp/ --warmup_epoch 80
# --is_eval --load_experiment relbert-webqsp-rearev-final.ckpt 

###ReaRev+LMSR training
# python main.py ReaRev  --entity_dim 64 --num_epoch 200 --batch_size 8 --eval_every 2  \
# --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name webqsp \
# --experiment_name prn_webqsp-rearev-lmsr  --data_folder data/webqsp/ --num_epoch 100 #--warmup_epoch 80


###Evaluate webqsp
python main.py ReaRev --entity_dim 64 --num_epoch 100 --batch_size 8 --eval_every 2 --data_folder data/webqsp/ --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3 --relation_word_emb True --load_experiment webqsp-rearev-relbert-res-psnr-h1.ckpt --is_eval --name webqsp --experiment_name webqsp-rearev-relbert-res-psnr