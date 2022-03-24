python run_clm.py \
    --config_name gpt2-small \
    --do_train \
    --do_eval \
    --num_train_epochs 55 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --fp16 \
    --warmup_ratio 0.01 \
    --output_dir ./ckpt/pretrain/${pt_name}/size_${size} \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 16 \
    --pt_dataset ./data/${pt_name}.pt \
    --dataset_limit_train ${size}000000 \
    --dataset_limit_valid 500000 \
    --dataset_limit_test 500000 \
    # --wandb 1

