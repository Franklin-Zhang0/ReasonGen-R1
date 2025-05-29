set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

SYSTEM_PROMPT=""

GPUS=`nvidia-smi -L | wc -l`
MODEL_PATH=Franklin0/ReasonGen-R1-SFT
RM_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
RUN_NAME=release_test
PROJ_NAME="verl_janus_test"
SAVE_DIR=ckpt/janus_rl/$PROJ_NAME/$RUN_NAME

TEMPLATE="A photo of {}. Output a richly detailed prompt: "



RM_TEMPLATE='You are given a text prompt: \"{prompt}\" 
Below is one generated image:
<image>

1. Describe the image thoroughly (objects, colors, layout, etc.), do not be affected by the prompt.
2. Identify key visual elements and instructions from the prompt.
3. Evaluate how well the image follows the prompt:
   - Are all required elements present?
   - Are object counts, colors, and positions accurate?

Be extremly strict and precise:
Only if the image matches the prompt perfectly, respond with: \\boxed{{1}}
Otherwise, respond with: \\boxed{{0}}

Reason before your final boxed answer. Only one number should appear inside the box.'

export HYDRA_FULL_ERROR=1

# if [ "$RANK" -eq 0 ]; then
python3 -m verl.trainer.image_generation_rl \
    algorithm.adv_estimator=grpo \
    data.train_files="[examples/rl_prompts/geneval_train.txt,examples/rl_prompts/dpg_train.txt,examples/rl_prompts/t2i_train.txt]" \
    data.val_files="[examples/rl_prompts/geneval_val.txt,examples/rl_prompts/dpg_val.txt,examples/rl_prompts/t2i_val.txt]" \
    data.num_val_samples=256 \
    data.system_prompt="$SYSTEM_PROMPT" \
    data.train_batch_size=8 \
    data.max_prompt_length=156 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    'data.prompt_template="'"$TEMPLATE"'"' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=-0.00 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.model.cfg_weight=1.0 \
    actor_rollout_ref.model.detach_uncond=True \
    actor_rollout_ref.rollout.micro_batch_size=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.cot_generate=True \
    actor_rollout_ref.actor.ignore_img_start=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.adaptive_entropy_coeff.enable=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=8 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    +trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.total_epochs=5 \
    trainer.max_steps=400 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$SAVE_DIR \
    reward_model.reward_manager=image_generation \
    reward_model.rollout.max_num_seqs=2 \
    reward_model.rollout.gpu_memory_utilization=0.5 \
    reward_model.model.path=$RM_MODEL_PATH \
    reward_model.micro_batch_size_per_gpu=8 \
    reward_model.paired=False \
    'reward_model.template="'"$RM_TEMPLATE"'"' \
    img_saving.save_freq=5 \
    img_saving.num=16 \
    reward_model.rollout.prompt_length=1536 \
    reward_model.rollout.max_num_batched_tokens=8192
