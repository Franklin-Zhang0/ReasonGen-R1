#!/bin/bash

all_model_name_list=(
    "Janus-Pro-7B"
    "Janus-Pro-7B-cot"
    "100k_sample_7B_bs128_lr2e-6_image_1.0_text_0.1_0429"
    "100k_sample_short_7B_bs128_lr2e-6_image_only_1.0_0429"
    "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_250"
    "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_150"
    "janus_image_only_dpo-0502_240"
    )

model_name_list=(
    # "Janus-Pro-7B"
    "Janus-Pro-7B-cot"
    # "100k_sample_7B_bs128_lr2e-6_image_1.0_text_0.1_0429"
    # "100k_sample_short_7B_bs128_lr2e-6_image_only_1.0_0429"
    # "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_250"
    # "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_150"
    # "janus_image_only_dpo-0502_240"
    )

for name in "${model_name_list[@]}"; do
    echo ""
    echo "Model: ${name}, inference start"
    echo ""
    /home/v-zhangyu3/miniforge3/envs/image_rl/bin/python generate_inference_geneval.py --model_name="${name}"
    echo "Model: ${name}, inference end"
    echo ""
    echo "======================"
    echo ""
    echo "Model: ${name}, evaluation start"
    echo ""
    /home/v-zhangyu3/miniforge3/envs/geneval/bin/python "/home/v-zhangyu3/project/geneval/evaluation/evaluate_images.py" \
        "/home/v-zhangyu3/project/Image-RL/Janus/geneval_out_result/geneval_output_${name}" \
        --outfile "geneval_out_result/results_output_${name}.jsonl" \
        --model-path "/home/v-zhangyu3/project/geneval/models"
    echo "Model: ${name}, evaluation end"
    echo ""
    echo "======================"
    echo ""
done

for name in "${all_model_name_list[@]}"; do
    echo ""
    echo "Result of ${name}"
    /home/v-zhangyu3/miniforge3/envs/geneval/bin/python /home/v-zhangyu3/project/geneval/evaluation/summary_scores.py "geneval_out_result/results_output_${name}.jsonl"
    echo "====================="
    echo ""
done

