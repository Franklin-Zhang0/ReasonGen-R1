set -x

TEMPLATE='A photo of {}. Generate a detailed description of how to create an image strictly based on the information in the caption. Do not add extra elements or creative interpretation beyond the raw caption. Pay close attention to all specific details in the caption—such as color, position, number, orientation, and object types. Your output should be a breakdown of how to create the image, suitable for guiding an image generation model. Please directly output the reasoning steps.'

nproc_per_node=4
save_path=/blob/franklin/ckpt/image_rl/janus_sft/100k_sample/100k_sample_7B_bs128_lr2e-6_image_only_1.0_0429
# save_path=/blob/franklin/ckpt/image_rl/janus_sft/test

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.img_rl_fsdp_sft_trainer \
    data.train_files=/blob/franklin/datasets/Janus_RL/sft_dataset/laion2b_aesthetic_set_100k@train \
    data.val_files=/blob/franklin/datasets/Janus_RL/sft_dataset/laion2b_aesthetic_set_100k@test \
    data.train_batch_size=256 \
    data.prompt_key=brief_caption \
    data.response_key=sft_prompt \
    data.max_length=1792 \
    data.micro_batch_size_per_gpu=4 \
    data.prompt_augmentation="[short_caption,paraphrases,tags,varied_captions,object_prompts]" \
    data.cot_augmentation="[detailed_caption,step_by_step,object_centric,tags,region_descriptions]" \
    'data.chat_template="'"$TEMPLATE"'"' \
    model.partial_pretrain=deepseek-ai/Janus-Pro-7B \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=sft_test \
    trainer.experiment_name=janus_sft \
    trainer.total_epochs=2 \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.wrap_policy.min_num_params=1000000 \
    trainer.logger=['console'] \
    optim.warmup_steps_ratio=0.1 \
    optim.lr=2e-6 \
    algorithm.loss_scale.image=1.0 \
    algorithm.loss_scale.text=0.0 \
    algorithm.loss_scale.image_start_token=0.0 \
    algorithm.loss_scale.gradual_increase_interval=[0.2,0.5] \
    algorithm.loss_scale.gradual_increase_key=text \
    algorithm.use_kl_loss=False \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_loss_weight=0.00 \
    algorithm.kl_loss_scale.image=1.0 \
    algorithm.kl_loss_scale.text=0.0 \
    algorithm.kl_loss_scale.image_start_token=0.0 \
    trainer.default_hdfs_dir=null $@

python ~/thinking.py > /dev/null 2>&1