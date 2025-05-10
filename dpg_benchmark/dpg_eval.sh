#!/bin/bash
set -x

export ACCELERATE_LOG_LEVEL=critical

HOME_PATH=$(cd ~ && pwd)
all_model_name_list=(
    "Janus-Pro-7B"
    "Janus-Pro-7B-cot"
    "100k_sample_7B_bs128_lr2e-6_image_1.0_text_0.1_0429_324"
    "100k_sample_short_7B_bs128_lr2e-6_image_only_1.0_0429"
    "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_250"
    "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_150"
    "janus_image_only_dpo-0502_240"
    "janus_image_only_dpo-eval_ds-0503_60"
    "janus_cot_dpo-0502-200"
    "image_only_grpo_4_rollout_40"
    "100k_sample_short_7B_bs128_lr1e-5_image_only_1.0-0501_297"
    "100k_sample_short_7B_bs128_lr2e-6_image_1.0_text_0.1-0430_297"
    "100k_sample_short_7B_bs128_lr2e-6_image_1.0_text_0.1_gradual0_0.5-0430_297"
    "100k_sample_short_7B_bs128_lr2e-6_image_only_1.0-0430_297"
    "100k_sample_short_7B_bs128_lr5e-6_image_1.0_text_0.1-0501_297"
    "100k_sample_short_7B_bs128_lr5e-6_image_1.0_text_0.5-0501_297"
    "100k_sample_short_7B_bs128_lr5e-6_image_only_1.0-0501_297"
    "100k_sample_7B_bs128_lr2e-6_image_only_1.0_0429_324"
    "100k_sample_7B_bs512_lr1e-5-loss_image_1.0_text_0.2_text_gradual-0428_324"
    "100k_sample_7B_bs512_lr1e-5-loss_image_1.0_text_0.5_text_gradual-0428_324"
    "100k_sample_7B_bs512_lr1e-5-loss_image_1.0_text_0.5_text_gradual-KL_0.1-0428_324"
    "100k_sample_7B_bs512_lr1e-5-loss_image_only_1.0-0428_648"
    "23k_sample_7B_bs128_lr1e-5-loss_scale_image_1.0_gradual_text1.0-0423_205"
    "23k_sample_7B_bs128_lr1e-5_Image1.0-Text1.0-Start0.0_prompt_template_0420_495"
    "100k_sample_short_7B_bs128_lr1e-5_image_only-0505_1990"
    "100k_sample_short_7B_bs128_lr1e-5_image_1.0_text_0.5-0505_1990"
    "200k_sample_short_7B_bs128_lr2e-5_image_1.0_text_0.5-0506"
    "200k_sample_short_7B_bs128_lr2e-5_image_1.0_text_0.2-0506"
    "image_only_grpo_8_rollout_bs32_mini16_cfg_1.0_no_kl_lr_5e-6_3_ds_400"
    )

model_name_list=(
    # "Janus-Pro-7B"
    # "Janus-Pro-7B-cot"
    # "100k_sample_7B_bs128_lr2e-6_image_1.0_text_0.1_0429_324"
    # "100k_sample_short_7B_bs128_lr2e-6_image_only_1.0_0429"
    # "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_250"
    # "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_150"
    # "janus_image_only_dpo-0502_240"
    # "janus_image_only_dpo-eval_ds-0503_60"
    # "janus_cot_dpo-0502_200"
    # "image_only_grpo_4_rollout_40"
    # "100k_sample_short_7B_bs128_lr1e-5_image_only_1.0-0501_297"
    # "100k_sample_short_7B_bs128_lr2e-6_image_1.0_text_0.1-0430_297"
    # "100k_sample_short_7B_bs128_lr2e-6_image_1.0_text_0.1_gradual0_0.5-0430_297"
    # "100k_sample_short_7B_bs128_lr2e-6_image_only_1.0-0430_297"
    # "100k_sample_short_7B_bs128_lr5e-6_image_1.0_text_0.1-0501_297"
    # "100k_sample_short_7B_bs128_lr5e-6_image_1.0_text_0.5-0501_297"
    # "100k_sample_short_7B_bs128_lr5e-6_image_only_1.0-0501_297"
    # "100k_sample_7B_bs128_lr2e-6_image_only_1.0_0429_324"
    # "100k_sample_7B_bs512_lr1e-5-loss_image_1.0_text_0.2_text_gradual-0428_324"
    # "100k_sample_7B_bs512_lr1e-5-loss_image_1.0_text_0.5_text_gradual-0428_324"
    # "100k_sample_7B_bs512_lr1e-5-loss_image_1.0_text_0.5_text_gradual-KL_0.1-0428_324"
    # "100k_sample_7B_bs512_lr1e-5-loss_image_only_1.0-0428_648"
    # "23k_sample_7B_bs128_lr1e-5-loss_scale_image_1.0_gradual_text1.0-0423_205"
    # "23k_sample_7B_bs128_lr1e-5_Image1.0-Text1.0-Start0.0_prompt_template_0420_495"
    # "100k_sample_short_7B_bs128_lr1e-5_image_only-0505_1990"
    # "100k_sample_short_7B_bs128_lr1e-5_image_1.0_text_0.5-0505_1990" # done eval
    # "200k_sample_short_7B_bs128_lr2e-5_image_1.0_text_0.5-0506_3320"
    # "200k_sample_short_7B_bs128_lr2e-5_image_1.0_text_0.2-0506_1660"
    # "200k_sample_short_7B_bs128_lr2e-5_image_1.0_text_0.2-0506_3320"

    # "image_only_grpo_8_rollout_bs32_mini16_cfg_1.0_no_kl_lr_5e-6_no_detach_strict_prompt_no_a_photo_of_180"
    # "image_only_grpo_8_rollout_bs32_mini16_cfg_1.0_no_kl_lr_5e-6_no_detach_strict_prompt_no_a_photo_of_100"
    # "image_only_grpo_8_rollout_bs32_mini16_cfg_1.0_no_kl_lr_5e-6_3_ds_400"


    "100k_sample_short_7B_bs128_lr1e-5_image_only-0505_1990_lora_398"



    # "image_only_grpo_8_rollout_kl_0.001_cfg_2.0_no_detach_140"
    # "image_only_grpo_8_rollout_kl_0.001_cfg_1.0_no_detach_no_a_photo_of_200"

    # "23k_sample_7B_bs128_lr1e-5_Image1.0-Text1.0-Start0.0_prompt_template_weight_decay0.0_0420_495" # half inference
    # "23k_sample_7B_bs128_lr2e-5-loss_image_0.1_other_1.0-KL0.1_image1.0_text0.1-0422_330"
    # "23k_sample_7B_bs128_lr2e-5-loss_no_image_other_1.0-KL0.1_text0.1_image1.0-0422_330"
    # "23k_sample_7B_bs128_lr2e-5-loss_no_text_other_1.0-KL0.1-0422_330"
    # "23k_sample_7B_bs128_lr2e-5-loss_scale_all_1.0-fix_loss_mask-0422_660"
    # "23k_sample_7B_bs128_lr2e-5-loss_scale_image_1.0_gradual_text0.2-kl_0.1_image_text-0423_660"
    # "23k_sample_7B_bs128_lr2e-5_0417_1320"
    # "23k_sample_7B_bs128_lr2e-5_Image1.0-Text0.5-Start10.0_prompt_template_0419_1320"
    # "23k_sample_7B_bs128_lr2e-5_Image1.0-Text1.0-Start0.0-L2_anchor_0420_660"
    # "23k_sample_7B_bs128_lr2e-5_Image1.0-Text1.0-Start2.0_0419_1320"
    # "23k_sample_7B_bs128_lr2e-5_loss-scaling_0419_1320"
    # "23k_sample_7B_bs512_lr1e-5-loss_scale_all_1.0-0423_205"
    # "23k_sample_7B_bs512_lr2e-5-loss_scale_all_1.0_text_gradual-0423_205"
    )

conda activate image_rl
for name in "${model_name_list[@]}"; do
    echo ""
    echo "Model: ${name}, inference start"
    echo ""
    accelerate launch --quiet $HOME_PATH/project/Image-RL/dpg_benchmark/generate_inference_dpg.py --model_name="${name}"
    CUDA_VISIBLE_DEVICES=0 python $HOME_PATH/project/Image-RL/dpg_benchmark/generate_inference_dpg.py --model_name="${name}"
    echo "Model: ${name}, inference end"
    echo ""
done

conda activate dpg_test
for name in "${model_name_list[@]}"; do
    echo ""
    echo "Model: ${name}, evaluation start"
    echo ""    
    mkdir -p "/blob/franklin/expdata/dpg_out_result/${name}"
    accelerate launch --num_machines 1 --num_processes 8 --multi_gpu --mixed_precision "fp16" \
    $HOME_PATH/project/ELLA/dpg_bench/compute_dpg_bench.py \
    --image-root-path "$HOME_PATH/project/Image-RL/dpg_benchmark/dpg_result/${name}/generated_images" \
    --res-path "/blob/franklin/expdata/dpg_out_result/${name}/eval_result.txt" \
    --csv "$HOME_PATH/project/ELLA/dpg_bench/dpg_bench.csv" \
    --resolution 384 \
    --pic-num 4 \
    --vqa-model mplug
done

# cnt=0
# conda activate geneval
# for name in "${model_name_list[@]}"; do
#     echo "======================"
#     echo ""
#     echo "Model: ${name}, evaluation start"
#     echo ""
#     CUDA_VISIBLE_DEVICES=$cnt python "$HOME_PATH/project/geneval/evaluation/evaluate_images.py" \
#         "$HOME_PATH/project/Image-RL/geneval_out_result/geneval_output_${name}" \
#         --outfile "/blob/franklin/expdata/geneval_out_result/results_output_${name}.jsonl" \
#         --model-path "$HOME_PATH/project/geneval/models" &
#     cnt=$((cnt + 1))
#     if [ $cnt -eq 2 ]; then
#         wait
#         cnt=0
#     fi
#     echo "Model: ${name}, evaluation end"
#     echo ""
#     echo "======================"
#     echo ""
# done

# conda activate geneval
# for name in "${all_model_name_list[@]}"; do
#     echo ""
#     echo "Result of ${name}"
#     python $HOME_PATH/project/geneval/evaluation/summary_scores.py "/blob/franklin/expdata/geneval_out_result/results_output_${name}.jsonl"
#     echo "====================="
#     echo ""
# done

python ~/thinking.py > /dev/null 2>&1
