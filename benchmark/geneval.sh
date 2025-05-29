#!/bin/bash
set -x

export ACCELERATE_LOG_LEVEL=critical

HOME_PATH=$(cd ~ && pwd)
all_model_name_list=(
    "Janus-Pro-7B"
    "ReasonGen-R1"
    )

model_name_list=(
    # "Janus-Pro-7B"
    "ReasonGen-R1"
    )

conda activate image_rl
for name in "${model_name_list[@]}"; do
    echo ""
    echo "Model: ${name}, inference start"
    echo ""
    accelerate launch --quiet $HOME_PATH/project/Image-RL/benchmark/generate_inference_geneval.py --model_name="${name}"
    CUDA_VISIBLE_DEVICES=0 python $HOME_PATH/project/Image-RL/benchmark/generate_inference_geneval.py --model_name="${name}"
    echo "Model: ${name}, inference end"
    echo ""
done

cnt=0
conda activate geneval
for name in "${model_name_list[@]}"; do
    echo "======================"
    echo ""
    echo "Model: ${name}, evaluation start"
    echo ""
    CUDA_VISIBLE_DEVICES=$cnt python "$HOME_PATH/project/geneval/evaluation/evaluate_images.py" \
        "$HOME_PATH/project/Image-RL/geneval_out_result/geneval_output_${name}" \
        --outfile "~/expdata/geneval_out_result/results_output_${name}.jsonl" \
        --model-path "$HOME_PATH/project/geneval/models" &
    cnt=$((cnt + 1))
    if [ $cnt -eq 1 ]; then
        wait
        cnt=0
    fi
    echo "Model: ${name}, evaluation end"
    echo ""
    echo "======================"
    echo ""
done

conda activate geneval
for name in "${all_model_name_list[@]}"; do
    echo ""
    echo "Result of ${name}"
    python $HOME_PATH/project/geneval/evaluation/summary_scores.py "~/expdata/geneval_out_result/results_output_${name}.jsonl"
    echo "====================="
    echo ""
done