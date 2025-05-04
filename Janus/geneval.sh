#!/bin/bash
set -x
HOME_PATH=$(cd ~ && pwd)
all_model_name_list=(
    "Janus-Pro-7B"
    "Janus-Pro-7B-cot"
    "100k_sample_7B_bs128_lr2e-6_image_1.0_text_0.1_0429"
    "100k_sample_short_7B_bs128_lr2e-6_image_only_1.0_0429"
    "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_250"
    "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_150"
    "janus_image_only_dpo-0502_240"
    "janus_image_only_dpo-eval_ds-0503-60"
    "janus_cot_dpo-0502-200"
     "image_only_grpo_4_rollout_40"
    )

model_name_list=(
    # "Janus-Pro-7B"
    # "Janus-Pro-7B-cot"
    # "100k_sample_7B_bs128_lr2e-6_image_1.0_text_0.1_0429"
    # "100k_sample_short_7B_bs128_lr2e-6_image_only_1.0_0429"
    # "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_250"
    # "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_150"
    # "janus_image_only_dpo-0502_240"
    # "janus_image_only_dpo-eval_ds-0503-60"
    "janus_cot_dpo-0502_200"
    "image_only_grpo_4_rollout_40"
    )

conda activate image_rl
for name in "${model_name_list[@]}"; do
    echo ""
    echo "Model: ${name}, inference start"
    echo ""
    accelerate launch $HOME_PATH/project/Image-RL/Janus/generate_inference_geneval.py --model_name="${name}"
    echo "Model: ${name}, inference end"
    echo ""
done

conda activate geneval
for name in "${model_name_list[@]}"; do
    echo "======================"
    echo ""
    echo "Model: ${name}, evaluation start"
    echo ""
    python "$HOME_PATH/project/geneval/evaluation/evaluate_images.py" \
        "$HOME_PATH/project/Image-RL/geneval_out_result/geneval_output_${name}" \
        --outfile "/blob/franklin/expdata/geneval_out_result/results_output_${name}.jsonl" \
        --model-path "$HOME_PATH/project/geneval/models"
    echo "Model: ${name}, evaluation end"
    echo ""
    echo "======================"
    echo ""
done

conda activate geneval
for name in "${all_model_name_list[@]}"; do
    echo ""
    echo "Result of ${name}"
    python $HOME_PATH/project/geneval/evaluation/summary_scores.py "/blob/franklin/expdata/geneval_out_result/results_output_${name}.jsonl"
    echo "====================="
    echo ""
done

python ~/thinking.py > /dev/null 2>&1
