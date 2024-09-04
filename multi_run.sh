torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_addr=127.0.0.1 --master_port 3000 multi_train.py \
--seed 22 \
--model_name "coatnet_2_rw_224.sw_in12k_ft_in1k" \
--img_size 224 \
--num_epoch 50 \
--t_batch_size 32 \
--v_batch_size 64 \
--lr 1e-4 \
--lr_scheduler "cosine" \
--min_lr 1e-6 \
--t_max 500 \
--w_decay 1e-6 \
--fold 0 \
--n_fold 5 \
--t_0 12 \
--n_accumulate 1 \
--quant True

