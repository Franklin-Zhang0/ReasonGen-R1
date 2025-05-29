set -x

TEMPLATE='{}. Output a richly detailed prompt: '

nproc_per_node=4
RUN_NAME="janus_sft"
save_path=ckpt

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.img_rl_fsdp_sft_trainer \
    data.train_files=Franklin0/ReasonGen-R1-SFT-230k@train \
    data.val_files=Franklin0/ReasonGen-R1-SFT-230k@test \
    data.train_batch_size=128 \
    data.prompt_key=brief_caption \
    data.response_key=detailed_caption \
    data.max_length=1280 \
    data.micro_batch_size_per_gpu=4 \
    data.prompt_augmentation=[short_caption,paraphrases,tags,varied_captions,object_prompts] \
    data.prompt_dropout=0.1 \
    'data.chat_template="'"$TEMPLATE"'"' \
    model.partial_pretrain=deepseek-ai/Janus-Pro-7B \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path/$RUN_NAME \
    trainer.project_name=sft_test \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=1 \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.wrap_policy.min_num_params=1000000 \
    trainer.logger=[console] \
    optim.warmup_steps_ratio=0.05 \
    optim.lr=1e-5 \
    algorithm.loss_scale.image=1.0 \
    algorithm.loss_scale.text=2.0 \
    algorithm.loss_scale.image_start_token=0.0 \
    algorithm.use_kl_loss=False \
    trainer.default_hdfs_dir=null