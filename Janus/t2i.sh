#!/bin/bash
# set -x

export ACCELERATE_LOG_LEVEL=critical

HOME_PATH=$(cd ~ && pwd)

model_name_list=(
    # "Janus-Pro-7B"
    "grpo_8_rollout_bs32_mini16_cfg_1.0_cot_lr_5e-6_3_ds_sft_2_batch_subsample_fixed_entropy_-0.001"
)

# setting_list=(
#     "texture"
#     "color"
#     "shape"
#     "spatial"
#     "non_spatial"
#     "numeracy"
#     "complex"
# )

setting_list=(
    "texture_val"
    "color_val"
    "shape_val"
    "spatial_val"
    "non_spatial_val"
    "numeracy_val"
    "complex_val"
)

# conda activate image_rl
# for name in "${model_name_list[@]}"; do
#     echo ""
#     echo "Model: ${name}, t2i inference start"
#     echo ""
#     for setting in "${setting_list[@]}"; do
#         echo "Inference setting: ${setting}"
#         echo ""
#         accelerate launch --quiet $HOME_PATH/project/Image-RL/Janus/generate_inference_t2i.py --model_name="${name}" --setting=${setting}
#         CUDA_VISIBLE_DEVICES=0 python $HOME_PATH/project/Image-RL/Janus/generate_inference_t2i.py --model_name="${name}" --setting=${setting}
#         echo ""
#         echo "Inference setting: ${setting} done"
#         echo ""
#     done
#     echo "Model: ${name}, all inference end"
#     echo ""
# done

conda activate t2i
data_path="$HOME_PATH/project/T2I-CompBench/examples/dataset/"
for name in "${model_name_list[@]}"; do
    # echo "======================"
    # echo ""
    # echo "Model: ${name}, start evaluating"
    # echo ""
    # cd $HOME_PATH/project/T2I-CompBench/BLIPvqa_eval
    # CUDA_VISIBLE_DEVICES=0 python $HOME_PATH/project/T2I-CompBench/BLIPvqa_eval/BLIP_vqa.py --out_dir="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}/texture_val" &
    # CUDA_VISIBLE_DEVICES=1 python $HOME_PATH/project/T2I-CompBench/BLIPvqa_eval/BLIP_vqa.py --out_dir="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}/color_val" &
    # CUDA_VISIBLE_DEVICES=2 python $HOME_PATH/project/T2I-CompBench/BLIPvqa_eval/BLIP_vqa.py --out_dir="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}/shape_val" &
    # CUDA_VISIBLE_DEVICES=3 python $HOME_PATH/project/T2I-CompBench/BLIPvqa_eval/BLIP_vqa.py --out_dir="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}/complex_val" & 
    # CUDA_VISIBLE_DEVICES=7 python $HOME_PATH/project/T2I-CompBench/CLIPScore_eval/CLIP_similarity.py --outpath="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}/complex_val" --complex True &
    # CUDA_VISIBLE_DEVICES=6 python $HOME_PATH/project/T2I-CompBench/CLIPScore_eval/CLIP_similarity.py --outpath="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}/non_spatial_val" &
    # wait
    # cd $HOME_PATH/project/T2I-CompBench/UniDet_eval
    # python $HOME_PATH/project/T2I-CompBench/UniDet_eval/2D_spatial_eval.py --outpath="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}/spatial_val"
    # python $HOME_PATH/project/T2I-CompBench/UniDet_eval/numeracy_eval.py --outpath="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}/numeracy_val"
    # python $HOME_PATH/project/T2I-CompBench/UniDet_eval/2D_spatial_eval.py --outpath="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}/complex_val" --complex True
    # python $HOME_PATH/project/T2I-CompBench/3_in_1_eval/3_in_1.py --outpath="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}/complex_val" --data_path=${data_path}
    # wait
    python /home/aiscuser/annie/show_t2i_result.py --folder_path="$HOME_PATH/project/T2I-CompBench/examples/outputs/${name}"
done

python ~/thinking.py > /dev/null 2>&1