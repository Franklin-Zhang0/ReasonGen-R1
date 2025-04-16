set -x

nproc_per_node=1
save_path=/home/v-zhangyu3/expdata/janus_sft/sft_test

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.img_rl_fsdp_sft_trainer \
    data.train_files=/blob/franklin/datasets/Janus_RL/sft_dataset/all_type@train \
    data.val_files=/blob/franklin/datasets/Janus_RL/sft_dataset/all_type@test \
    data.train_batch_size=256 \
    data.prompt_key=brief_caption \
    data.response_key=sft_prompt \
    data.max_length=1536 \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=deepseek-ai/Janus-Pro-1B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=sft_test \
    trainer.experiment_name=janus_sft \
    trainer.total_epochs=4 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@