set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

SYSTEM_PROMPT="Reason about how you should generate the image"

GPUS=`nvidia-smi -L | wc -l`
MODEL_PATH=deepseek-ai/Janus-Pro-7B  # replace it with your local file path
RM_MODEL_PATH=Qwen/Qwen2.5-VL-32B-Instruct
RUN_NAME="Janus_pro_7B-DPO-filter-Qwen32b-NO_KL"
PROJ_NAME="verl_janus_test"
SAVE_DIR=/blob/franklin/ckpt/image_rl/$PROJ_NAME/$RUN_NAME
export HYDRA_FULL_ERROR=1

if [ "$RANK" -eq 0 ]; then
python3 -m verl.trainer.image_generation_rl \
    algorithm.adv_estimator=dpo \
    data.train_files=/blob/franklin/datasets/Janus_RL/geneval/prompts/generation_prompts.txt \
    data.val_files=/blob/franklin/datasets/Janus_RL/geneval/prompts/generation_prompts.txt \
    data.system_prompt="$SYSTEM_PROMPT" \
    data.train_batch_size=32 \
    data.max_prompt_length=128 \
    data.max_response_length=600 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.model.cfg_weight=5.0 \
    actor_rollout_ref.model.detach_uncond=True \
    actor_rollout_ref.rollout.micro_batch_size=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.filter_groups.enable=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$SAVE_DIR \
    reward_model.reward_manager=image_generation \
    reward_model.model.path=$RM_MODEL_PATH \
    reward_model.max_num_batched_tokens=16384 \
    reward_model.micro_batch_size_per_gpu=16 \
    reward_model.rollout.tensor_model_parallel_size=4 \
    img_saving.save_freq=5 \
    img_saving.num=16
    
python ~/thinking.py > /dev/null 2>&1

fi
