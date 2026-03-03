uv run python -m rl.train_reward_model \
    --tag balanced_dataset_v1 \
    --batch_size 128 \
    --num_epochs 2 \
    --lr 1e-5 \
    --lora_rank 16 \
    --report_to wandb \
    --hf_repo MRBSTUDIO/humor-qwen3-8b \
    --hf_path_in_repo reward_model \
    --early_stopping_patience 7    