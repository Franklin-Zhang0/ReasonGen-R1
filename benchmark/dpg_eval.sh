#!/bin/bash
set -x

export ACCELERATE_LOG_LEVEL=critical

HOME_PATH=$(cd ~ && pwd)
all_model_name_list=(
    "Janus-Pro-7B"
    "ReasonGen-R1"
    )

model_name_list=(
    "Janus-Pro-7B"
    "ReasonGen-R1"
    )

conda activate image_rl
for name in "${model_name_list[@]}"; do
    echo ""
    echo "Model: ${name}, inference start"
    echo ""
    accelerate launch --quiet $HOME_PATH/project/Image-RL/benchmark/generate_inference_dpg.py --model_name="${name}"
    CUDA_VISIBLE_DEVICES=0 python $HOME_PATH/project/Image-RL/benchmark/generate_inference_dpg.py --model_name="${name}"
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
    --image-root-path "$HOME_PATH/project/Image-RL/benchmark/dpg_result/${name}/generated_images" \
    --res-path "~/expdata/dpg_out_result/${name}/eval_result.txt" \
    --csv "$HOME_PATH/project/ELLA/dpg_bench/dpg_bench.csv" \
    --resolution 384 \
    --pic-num 4 \
    --vqa-model mplug
done
