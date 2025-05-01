set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

SYSTEM_PROMPT=""

GPUS=`nvidia-smi -L | wc -l`
MODEL_PATH=deepseek-ai/Janus-Pro-7B  # replace it with your local file path
RM_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
RUN_NAME="Janus_pro_7B-DPO-filter"
PROJ_NAME="verl_janus_test"
SAVE_DIR=/blob/franklin/ckpt/image_rl/$PROJ_NAME/$RUN_NAME
TEMPLATE='A photo of {}. Generate a detailed description of how to create an image strictly based on the information in the caption. Do not add extra elements or creative interpretation beyond the raw caption. Pay close attention to all specific details in the caption—such as color, position, number, orientation, and object types. Your output should be a breakdown of how to create the image, suitable for guiding an image generation model. Please directly output the reasoning steps.'
RM_TEMPLATE='You are given a textual prompt: \"{prompt}\"
Below are two images generated from this prompt:
<image>   <image>

1. Identify the key visual elements described in the prompt.
2. For each element, check whether it appears perfectly in Image 1 and in Image 2.
3. If exactly one image matches the prompt perfectly, respond with \boxed{{1}} for Image 1 or \boxed{{2}} for Image 2.
4. If no image matches all elements perfectly, or if both images match equally well, respond with \boxed{{-1}}.

Provide your step-by-step reasoning first, then your final boxed answer. Only one number should appear inside the box.
'

export HYDRA_FULL_ERROR=1

if [ "$RANK" -eq 0 ]; then
python3 -m verl.trainer.image_generation_rl \
    algorithm.adv_estimator=dpo \
    data.train_files=/blob/franklin/datasets/Janus_RL/geneval/prompts/generation_prompts.txt \
    data.val_files=/blob/franklin/datasets/Janus_RL/geneval/prompts/generation_prompts.txt \
    data.system_prompt="$SYSTEM_PROMPT" \
    data.train_batch_size=16 \
    data.max_prompt_length=128 \
    data.max_response_length=600 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    'data.prompt_template="'"$TEMPLATE"'"' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
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
    actor_rollout_ref.rollout.cot_generate=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.filter_groups.enable=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$SAVE_DIR \
    reward_model.reward_manager=image_generation \
    reward_model.model.path=$RM_MODEL_PATH \
    reward_model.micro_batch_size_per_gpu=16 \
    'reward_model.template="'"$RM_TEMPLATE"'"' \
    img_saving.save_freq=5 \
    img_saving.num=16
    
python ~/thinking.py > /dev/null 2>&1

fi
