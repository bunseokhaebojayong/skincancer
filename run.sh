python train.py \
--seed 22 \
--model_name "swin_large_patch4_window7_224.ms_in22k_ft_in1k" \
--img_size 224 \
--num_epoch 50 \
--t_batch_size 64 \
--v_batch_size 64 \
--lr 1e-4 \
--lr_scheduler "cosine" \
--min_lr 1e-6 \
--t_max 500 \
--w_decay 1e-6 \
--fold 0 \
--n_fold 5 \
--t_0 12 \
--n_accumulate 1

