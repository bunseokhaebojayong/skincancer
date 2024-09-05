accelerator launch train.py \
--seed 22 \
--model_name "tf_efficientnetv2_m.in21k" \
--img_size 224 \
--num_epoch 50 \
--t_batch_size 8 \
--v_batch_size 16 \
--lr 1e-4 \
--lr_scheduler "cosine" \
--min_lr 1e-6 \
--t_max 500 \
--w_decay 1e-6 \
--fold 0 \
--n_fold 5 \
--t_0 12 \
--n_accumulate 4

