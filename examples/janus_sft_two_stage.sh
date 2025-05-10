set -x

TEMPLATE='{}. '

nproc_per_node=`nvidia-smi -L | wc -l`
save_path=/blob/franklin/ckpt/image_rl/janus_sft/200k_sample_aug_long/200k_sample_aug_long_7B_bs128_lr1e-5_all_1.0-two_stage-0510
# save_path=/blob/franklin/ckpt/image_rl/janus_sft/test

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.img_rl_fsdp_sft_trainer \
    data.train_files=/blob/franklin/datasets/Janus_RL/sft_dataset/short_brief_laion_aesthetic_200k_prompt_aug_long@train \
    data.val_files=/blob/franklin/datasets/Janus_RL/sft_dataset/short_brief_laion_aesthetic_200k_prompt_aug_long@test \
    data.train_batch_size=128 \
    data.prompt_key=brief_caption \
    data.response_key=detailed_caption \
    data.prompt_augmentation="[short_caption,paraphrases,tags,varied_captions,object_prompts]" \
    data.max_length=1280 \
    data.micro_batch_size_per_gpu=2 \
    'data.chat_template="'"$TEMPLATE"'"' \
    model.partial_pretrain=deepseek-ai/Janus-Pro-7B \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=sft_test \
    trainer.experiment_name=200k_sample_aug_long_7B_bs128_lr1e-5_all_1.0-two_stage-0510 \
    trainer.total_epochs=1 \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.wrap_policy.min_num_params=1000000 \
    trainer.logger=['console','wandb'] \
    optim.warmup_steps_ratio=0.05 \
    optim.lr=1e-5 \
    algorithm.loss_scale.image=1.0 \
    algorithm.loss_scale.text=1.0 \
    algorithm.loss_scale.image_start_token=1.0 \
    algorithm.two_stage=True \
    trainer.default_hdfs_dir=null $@

python ~/thinking.py > /dev/null 2>&1