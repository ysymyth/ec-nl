python run_clm.py \
    --config_name gpt2-small \
    --do_train \
    --do_eval \
    --num_train_epochs 12 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 10000000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --fp16 \
    --warmup_steps 1000 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --pt_dataset ./data/${ft_name}.pt \
    --dataset_limit_train 2000000 \
    --model_name_or_path ./ckpt/pretrain/${pt_name}/size_${size}/checkpoint-${ckpt} \
    --output_dir ./ckpt/finetune/${pt_name}/size_${size}/checkpoint-${ckpt}/${ft_name} \
    --overwrite_output_dir \
    --do_predict \
    --prediction_loss_only \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --scheduler_steps 25000 \
    # --wandb 1