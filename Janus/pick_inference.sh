#!/bin/bash
set -x

export ACCELERATE_LOG_LEVEL=critical

HOME_PATH=$(cd ~ && pwd)

# model_name_list=(
#     )


conda activate image_rl
echo "======================"
echo ""
echo "Start geneval inference"
echo ""
name="image_only_grpo_8_rollout_bs32_mini16_cfg_1.0_cot_no_kl_lr_5e-6_3_ds_sft2_batch_subsample_adaptive_entropy_max_coeff_3e-3_seperate_target_text_2.0_img_6.5_225"
accelerate launch --quiet $HOME_PATH/project/Image-RL/Janus/generate_inference_geneval_with_out_arg.py --model_name="${name}" --out_dir="$HOME_PATH/annie/geneval_output"
CUDA_VISIBLE_DEVICES=0 python $HOME_PATH/project/Image-RL/Janus/generate_inference_geneval_with_out_arg.py --model_name="${name}" --out_dir="$HOME_PATH/annie/geneval_output" 
echo ""
echo "End geneval inference"
echo ""
echo "======================="
echo ""
echo "Start geneval evaluation"
echo ""

conda activate geneval
python "$HOME_PATH/project/geneval/evaluation/evaluate_images.py" \
    "$HOME_PATH/annie/geneval_output" \
    --outfile "$HOME_PATH/annie/geneval_results_output.jsonl" \
    --model-path "$HOME_PATH/project/geneval/models"
echo ""
echo "End geneval evaluation"
echo ""

conda activate image_rl
echo "======================"
echo ""
echo "Start train_dpg inference"
echo ""
accelerate launch --quiet $HOME_PATH/project/Image-RL/Janus/generate_inference_txt_out.py --model_name="${name}" --out_dir="$HOME_PATH/annie/train_dpg_pick_inference_out" --txt_path="/blob/franklin/datasets/Janus_RL/dpg_prompts/train_dpg.txt"
CUDA_VISIBLE_DEVICES=0 python $HOME_PATH/project/Image-RL/Janus/generate_inference_txt_out.py --model_name="${name}" --out_dir="$HOME_PATH/annie/train_dpg_pick_inference_out" --txt_path="/blob/franklin/datasets/Janus_RL/dpg_prompts/train_dpg.txt"
echo ""
echo "End train_dpg inference"
echo ""
echo "======================="
echo ""
echo "Start merged_dpg inference"
echo ""
accelerate launch --quiet $HOME_PATH/project/Image-RL/Janus/generate_inference_txt_out.py --model_name="${name}" --out_dir="$HOME_PATH/annie/merged_dpg_pick_inference_out" --txt_path="/blob/franklin/datasets/Janus_RL/dpg_prompts/merged_dpg.txt"
CUDA_VISIBLE_DEVICES=0 python $HOME_PATH/project/Image-RL/Janus/generate_inference_txt_out.py --model_name="${name}" --out_dir="$HOME_PATH/annie/merged_dpg_pick_inference_out" --txt_path="/blob/franklin/datasets/Janus_RL/dpg_prompts/merged_dpg.txt"
echo ""
echo "End merged_dpg inference"
echo ""
echo "======================="
echo ""
echo "Start merged_t2i inference"
echo ""
accelerate launch --quiet $HOME_PATH/project/Image-RL/Janus/generate_inference_txt_out.py --model_name="${name}" --out_dir="$HOME_PATH/annie/merged_t2i_pick_inference_out" --txt_path="/blob/franklin/datasets/Janus_RL/t2i_prompts/merged_t2i_train.txt"
CUDA_VISIBLE_DEVICES=0 python $HOME_PATH/project/Image-RL/Janus/generate_inference_txt_out.py --model_name="${name}" --out_dir="$HOME_PATH/annie/merged_t2i_pick_inference_out" --txt_path="/blob/franklin/datasets/Janus_RL/t2i_prompts/merged_t2i_train.txt"
echo ""
echo "End merged_t2i inference"
echo ""

python ~/thinking.py > /dev/null 2>&1
