set -x

TEMPLATE='A photo of {}. Generate a detailed description of how to create an image strictly based on the information in the caption. Do not add extra elements or creative interpretation beyond the raw caption. Pay close attention to all specific details in the caption—such as color, position, number, orientation, and object types. Your output should be a breakdown of how to create the image, suitable for guiding an image generation model. Please directly output the reasoning steps.'

nproc_per_node=4
save_path=/blob/franklin/ckpt/image_rl/janus_sft/100k_sample_short/100k_sample_short_7B_bs128_lr1e-5_image_only-0505/text_lora
# save_path=/blob/franklin/ckpt/image_rl/janus_sft/test

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.img_rl_fsdp_sft_trainer \
    data.train_files=/blob/franklin/datasets/Janus_RL/sft_dataset/short_brief_laion_aesthetic_100k@train \
    data.val_files=/blob/franklin/datasets/Janus_RL/sft_dataset/short_brief_laion_aesthetic_100k@test \
    data.train_batch_size=128 \
    data.prompt_key=brief_caption \
    data.response_key=sft_prompt \
    data.max_length=1792 \
    data.micro_batch_size_per_gpu=2 \
    'data.chat_template="'"$TEMPLATE"'"' \
    model.partial_pretrain=/blob/franklin/ckpt/image_rl/janus_sft/100k_sample_short/100k_sample_short_7B_bs128_lr1e-5_image_only-0505/global_step_1990 \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=sft_test \
    trainer.experiment_name=100k_sample_short_7B_bs128_lr1e-5_image_only-0505-text_lora \
    trainer.total_epochs=5 \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.wrap_policy.min_num_params=1000000 \
    model.lora_rank=128 \
    model.lora_alpha=128 \
    trainer.logger=['console','wandb'] \
    optim.warmup_steps_ratio=0.05 \
    optim.lr=3e-4 \
    algorithm.loss_scale.image=0.0 \
    algorithm.loss_scale.text=1.0 \
    algorithm.loss_scale.image_start_token=1.0 \
    algorithm.use_kl_loss=False \
    trainer.default_hdfs_dir=null $@

python ~/thinking.py > /dev/null 2>&1