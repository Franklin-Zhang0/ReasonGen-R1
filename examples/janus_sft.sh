set -x

nproc_per_node=1
save_path=/blob/franklin/ckpt/image_rl/janus_sft/23k_sample_7B_bs128_lr2e-5_loss-scaling_0419
# save_path=/blob/franklin/ckpt/image_rl/janus_sft/test


torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.img_rl_fsdp_sft_trainer \
    data.train_files=/blob/franklin/datasets/Janus_RL/sft_dataset/all_type@train \
    data.val_files=/blob/franklin/datasets/Janus_RL/sft_dataset/all_type@test \
    data.train_batch_size=128 \
    data.prompt_key=brief_caption \
    data.response_key=sft_prompt \
    data.max_length=2048 \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=deepseek-ai/Janus-Pro-7B \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=sft_test \
    trainer.experiment_name=janus_sft \
    trainer.total_epochs=8 \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.wrap_policy.min_num_params=1000000 \
    trainer.logger=['console'] \
    optim.warmup_steps_ratio=0.05 \
    optim.lr=2e-5 \
    algorithm.loss_scale.image=1.0 \
    algorithm.loss_scale.text=0.1 \
    algorithm.loss_scale.image_start_token=2.0 \
    trainer.default_hdfs_dir=null $@

python ~/thinking.py > /dev/null 2>&1