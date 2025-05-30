<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->


<div align="center">
<h1> ReasonGen-R1: <img src="./images/favicon.ico" alt="ReasonGen-R1 Logo" width="48" height="48"><br> Cot for Autoregressive Image generation models through SFT and RL</h1>


</div>

<div align="center">

  <a href="https://aka.ms/reasongen" target="_blank">
    <img alt="Homepage" src="https://img.shields.io/badge/HomePage-blue" />
  </a>
  </a>
  <a href="https://huggingface.co/collections/Franklin0/reasongen-r1-6836ed61fc4f6db543c0d368" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ReasonGen%20R1-ffc107?color=ffc107&logoColor=white" />
  </a>

</div>




<p align="center">
  <a href="#2-model-download"><b>📥 Model Download</b></a> |
  <a href="#3-quick-start"><b>⚡ Quick Start</b></a> |
  <a href="#4-acknowledgements"><b>📜 Acknowledgement</b></a> |
  <a href="#5-citation"><b>📖 Citation</b></a> <br>
  📄 <a href="xxxxxx"><b>Paper Link</b></a>
</p>

<div align="center">
<img alt="image" src="images/benchmark_and_comparison_white_bg.png" style="width:90%; margin-top: 10px;">
</div>

## 1. Introduction

Although chain-of-thought (CoT) reasoning and reinforcement learning (RL) have driven breakthroughs in NLP, their integration into generative vision models remains underexplored. We introduce ReasonGen-R1, a two-stage framework that first imbues an autoregressive image generator with explicit text-based "thinking" skills via supervised fine-tuning (SFT) on a newly generated reasoning dataset of written rationales, and then refines its outputs using Group Relative Policy Optimization (GRPO).
To enable the model to reason through text before generating images, We automatically generate and release a corpus of model-crafted rationales paired with visual prompts, enabling controlled planning of object layouts, styles, and scene compositions.
Our GRPO algorithm uses reward signals from a pretrained vision–language model to assess overall visual quality, optimizing the policy in each update.
Evaluations on Geneval, DPG, and the T2I benchmark demonstrate that ReasonGen-R1 consistently outperforms strong baselines and prior state-of-the-art models. We will open-source our generated reasoning dataset and training code to accelerate further advances in text-based reasoning–driven image generation. 

<div align="center">
<img alt="image" src="images/model_structure_white_bg.png" style="width:90%;">
</div>
 

## 2. Model Download
### Huggingface

| Model                 | Download                                                                    |
|-----------------------|-----------------------------------------------------------------------------|
| ReasonGen-R1 | [🤗 Hugging Face](https://huggingface.co/Franklin0/ReasonGen-R1) |
| ReasonGen-R1-SFT-Only | [🤗 Hugging Face](https://huggingface.co/Franklin0/ReasonGen-R1-SFT) |



## 3. Quick Start

### Installation

You can install the necessary dependencies by running the following command:

```shell
cd ~
mkdir project
cd project
conda create -n image_rl python==3.12 -y
conda activate image_rl
pip3 install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
git clone https://github.com/Franklin-Zhang0/ReasonGen-R1.git
cd Image-RL
pip install -r requirements.txt
pip install -e .
pip install -e ./Janus
```

<details>
<summary><h3>Evaluation Environment Installation (Optional)</h3></summary>
If you want to run the evaluation code, you can install the evaluation environment by running the following commands:

```shell
# Geneval
cd ~
mkdir project
cd project
git clone https://github.com/djghosh13/geneval.git
cd geneval
conda deactivate
conda create -n geneval python=3.9 -y
conda activate geneval
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmengine==0.7.3

pip install pandas
pip install numpy==1.23.1

pip install open-clip-torch
pip install clip-benchmark

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .

cd ../
bash ./evaluation/download_models.sh "./models"
```

```shell
# DPG
cd ~
cd project
git clone https://github.com/TencentQQGYLab/ELLA.git
cd ELLA
cp ~/project/ReasonGen-R1/requirements-for-dpg_bench.txt .
conda deactivate
conda create -n dpg_test python=3.9 -y
conda activate dpg_test
conda install conda-forge::fairseq -y
pip install -r requirements-for-dpg_bench.txt
```

Once the eval environment is setup, you can use the following commands to run the evaluation:
```shell
bash -i benchmark/geneval.sh
bash -i benchmark/dpg_eval.sh
```
</details>

### Inference
To inference with the ReasonGen-R1 model, you can use the following command:
```shell
python Image-RL/Janus/cot_generate_inference.py
```

### SFT Training
To train the SFT model from Janus-Pro-7B model on the ReasonGen-R1-SFT-200k dataset, you can use the following command:
```shell
bash Image-RL/examples/janus_sft.sh
```

### RL Training
To train the RL model from the ReasonGen-R1-SFT model, you can use the following command:
```shell
bash Image-RL/Janus/janus_rl.py
```


## 4. Acknowledgements
We would like to thank <a href="https://github.com/volcengine/verl">Verl</a>, upon which our repo is built.

## 5. Citation

```bibtex
@article{yu2025reasongen,
  title={ReasonGen-R1: Cot for Autoregressive Image generation models through SFT and RL},
  author={Yu Zhang, Yunqi Li, Yifan Yang, Rui Wang， Yuqin Yang, Qi Dai, Jianming Bao, Dongdong Chen, Chong Luo, Lili Qiu},
  year={2025}
}
```
