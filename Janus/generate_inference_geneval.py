import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import json
from tqdm import tqdm
import tyro

from peft import PeftModel
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
from accelerate import Accelerator
accelerator = Accelerator()


# specify the path to the model

available_models={
    "Janus-Pro-7B": {"model_path": "deepseek-ai/Janus-Pro-7B", "use_cot": False},
    "Janus-Pro-7B-cot": {"model_path": "deepseek-ai/Janus-Pro-7B", "use_cot": True},
    "100k_sample_7B_bs128_lr2e-6_image_1.0_text_0.1_0429_324":{
        "model_path": "/blob/franklin/ckpt/image_rl/janus_sft/100k_sample/100k_sample_7B_bs128_lr2e-6_image_1.0_text_0.1_0429/global_step_324/",
        "use_cot": True
    },
    "100k_sample_short_7B_bs128_lr2e-6_image_only_1.0_0429":{
        "model_path":"/blob/franklin/ckpt/image_rl/janus_sft/100k_sample_short/100k_sample_short_7B_bs128_lr2e-6_image_only_1.0_0429/global_step_99/",
        "use_cot": True
    },
    "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_250": {
        "model_path": "/blob/franklin/ckpt/image_rl/verl_janus_test/Janus_pro_7B-DPO-filter-16k_data-A_prompt/global_step_250/actor/huggingface",
        "use_cot": False
    },
    "Janus_pro_7B-DPO-filter-16k_data-A_prompt_step_150":{
        "model_path": "/blob/franklin/ckpt/image_rl/verl_janus_test/Janus_pro_7B-DPO-filter-16k_data-A_prompt/global_step_150/actor/huggingface",
        "use_cot": False
    },
    "janus_image_only_dpo-0502_240":{"model_path":"/blob/franklin/ckpt/image_rl/verl_janus_test/janus_image_only_dpo-0502/global_step_240/actor/huggingface","use_cot": False},
    "janus_image_only_dpo-eval_ds-0503_60":{"model_path":"/blob/franklin/ckpt/image_rl/verl_janus_test/janus_image_only_dpo-eval_ds-0503/global_step_60/actor/huggingface","use_cot": False},
    "janus_cot_dpo-0502_200":{
        "model_path":"/blob/franklin/ckpt/image_rl/verl_janus_test/janus_cot_dpo-0502/global_step_200/actor/huggingface",
        "use_cot": True
    },
    "image_only_grpo_4_rollout_40": {
        "model_path":"/blob/franklin/ckpt/image_rl/verl_janus_test/image_only_grpo_4_rollout/global_step_40/actor/huggingface",
        "use_cot": False
    },
    "100k_sample_short_7B_bs128_lr1e-5_image_only_1.0-0501_297":{
        "model_path":"/blob/franklin/ckpt/image_rl/janus_sft/100k_sample_short/100k_sample_short_7B_bs128_lr1e-5_image_only_1.0-0501/global_step_297/",
        "use_cot": True
    },
    "image_only_grpo_8_rollout_bs32_mini16_cfg_1.0_no_kl_lr_5e-6_no_detach_strict_prompt_no_a_photo_of_180":{
        "model_path":"/blob/franklin/ckpt/image_rl/verl_janus_test/image_only_grpo_8_rollout_bs32_mini16_cfg_1.0_no_kl_lr_5e-6_no_detach_strict_prompt_no_a_photo_of/global_step_180/actor/huggingface",
        "use_cot": False
    },
    "image_only_grpo_8_rollout_bs32_mini16_cfg_1.0_no_kl_lr_5e-6_no_detach_strict_prompt_no_a_photo_of_100":{
        "model_path":"/blob/franklin/ckpt/image_rl/verl_janus_test/image_only_grpo_8_rollout_bs32_mini16_cfg_1.0_no_kl_lr_5e-6_no_detach_strict_prompt_no_a_photo_of/global_step_100/actor/huggingface",
        "use_cot": False
    },
    "image_only_grpo_8_rollout_kl_0.001_cfg_2.0_no_detach_140":{
        "model_path":"/blob/franklin/ckpt/image_rl/verl_janus_test/image_only_grpo_8_rollout_kl_0.001_cfg_2.0_no_detach/global_step_140/actor/huggingface"
        ,"use_cot": False
    },
    "image_only_grpo_8_rollout_kl_0.001_cfg_1.0_no_detach_no_a_photo_of_200":{
        "model_path":"/blob/franklin/ckpt/image_rl\\verl_janus_test/image_only_grpo_8_rollout_kl_0.001_cfg_1.0_no_detach_no_a_photo_of/global_step_200/actor/huggingface",
        "use_cot": False
    }
}

all_new_models_path = "/blob/franklin/ckpt/image_rl/janus_sft/"
name_list = ["100k_sample_short", "100k_sample","23k_sample","200k_sample_short"]
for name in name_list:
    models_list = os.listdir(os.path.join(all_new_models_path, name))
    for model in models_list:
        steps_list = os.listdir(os.path.join(all_new_models_path, name, model))
        steps = [int(step.split("_")[-1]) for step in steps_list if step.startswith("global_step_")]
        steps.sort()
        max_step = steps[-1]
        model_path = os.path.join(all_new_models_path, name, model, f"global_step_{max_step}/")
        model_name = f"{model}_{max_step}"
        if model_name in available_models:
            if available_models[model_name]["model_path"] != model_path:
                print(f"Model name conflict: {model_name}")
                print("now model path is: ", model_path)
                exit(1)
            assert available_models[model_name]["use_cot"] == True
        available_models[model_name] = {
            "model_path": model_path,
            "use_cot": True
        }
        # print(f"\"{model_name}\"")

available_models['100k_sample_short_7B_bs128_lr1e-5_image_only-0505_1990_lora_398']={
    "model_path":"/blob/franklin/ckpt/image_rl/janus_sft/100k_sample_short/100k_sample_short_7B_bs128_lr1e-5_image_only-0505/global_step_1990",
    "lora_path":"/blob/franklin/ckpt/image_rl/janus_sft/100k_sample_short/100k_sample_short_7B_bs128_lr1e-5_image_only-0505/text_lora/global_step_398",
    "use_cot": True
}

# get tyro arguments
def get_args():
    from dataclasses import dataclass

    @dataclass
    class Args:
        model_name: str

    return tyro.cli(Args)
args = get_args()
model_name = args.model_name
out_dir = os.path.expanduser(f"~/project/Image-RL/geneval_out_result/geneval_output_{model_name}")
model_path = available_models[model_name]["model_path"]
use_cot = available_models[model_name]["use_cot"]
use_lora = hasattr(available_models[model_name], "lora_path")
if use_lora:
    lora_path = available_models[model_name]["lora_path"]
processor_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(processor_path)
tokenizer = vl_chat_processor.tokenizer

# model_path = "/blob/franklin/ckpt/image_rl/janus_sft/100k_sample/100k_sample_7B_bs512_lr1e-5-loss_image_only_1.0-0428/global_step_324"
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
if use_lora:
    lora_weights = PeftModel.from_pretrained(
        deepcopy(vl_gpt.language_model),
        lora_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        is_trainable=False,
    )
vl_gpt = vl_gpt.to(torch.bfloat16).eval().to(accelerator.device)


cot_assistant = """
1. Scene Layout:
    Begin by creating a flat horizontal surface to represent the table.
    Place a plate on top of the table, positioned in the center or a clearly visible part of the image.

2. Plate Configuration:
    The plate should be large enough to comfortably hold three fruits (1 apple, 2 bananas).
    Ensure that the plate is clearly visible, with no fruit or table obscuring its outline.

3. Fruit Placement:
    Place one apple in the plate. It should be: Whole and unpeeled.
    Clearly distinguishable in shape and texture from the bananas.
    
    Place two bananas in the plate. They should be:
    Whole, unpeeled, and side-by-side or slightly curved around the apple.
    Ensure all three fruits are entirely within the plate.

4. Perspective and Visibility:
    Choose a camera/viewing angle that clearly shows:
    All three fruits inside the plate.
    Enough of the table surface to confirm the plate is placed on it.
    All objects must be unobstructed and clearly identifiable.

5. Lighting and Focus:
    Use neutral or natural lighting to avoid color distortion.
    Ensure the apple and bananas are in clear focus—they are the subjects of the image."""

template = "A photo of {}. Generate a detailed description of how to create an image strictly based on the information in the caption. Do not add extra elements or creative interpretation beyond the raw caption. Pay close attention to all specific details in the caption—such as color, position, number, orientation, and object types. Your output should be a breakdown of how to create the image, suitable for guiding an image generation model. Please directly output the reasoning steps."

def get_prompt(text, cot = False):
    text = text.replace("A photo of", "").replace("a photo of", "").strip() # avoid redundant a photo of
    if cot:
        conversation = [
            {
                "role": "<|User|>",
                "content": template.format(text),
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        conversation = [
            {
                "role": "<|User|>",
                "content": "A photo of {}".format(text),
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    if not cot:
        prompt = sft_format + vl_chat_processor.image_start_tag
    else:
        prompt = sft_format
    return prompt

@torch.inference_mode()
def generate_from_geneval_jsonl(
    jsonl_path: str,
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    # prompt: str,
    out_dir: str,
    temperature: float = 1,
    parallel_size: int = 4,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    cot: bool = False,
):  
    def inference_from_prompt_cot(prompt):
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).cuda()
        attention_mask = torch.ones((len(input_ids)), dtype=torch.bool).cuda()
        do_sample = True
        max_new_tokens = 2048
        eos_token_id = vl_chat_processor.tokenizer.eos_token_id
        pad_token_id = vl_chat_processor.tokenizer.pad_token_id
        image_start_token_id = vl_chat_processor.image_start_id
        input_ids = input_ids.unsqueeze(0).repeat(parallel_size, 1)
        attention_mask = attention_mask.unsqueeze(0).repeat(parallel_size, 1)
        generation_config = {'cfg_weight': 5.0}
        if use_lora:
            output = vl_gpt.text_img_generate(
                input_ids,
                attention_mask, 
                do_sample, 
                max_new_tokens, 
                eos_token_id, 
                pad_token_id, 
                image_start_token_id,
                generation_config,
                text_lora_module=lora_weights,
            )
        else:
            output = vl_gpt.text_img_generate(
                input_ids,
                attention_mask, 
                do_sample, 
                max_new_tokens, 
                eos_token_id, 
                pad_token_id, 
                image_start_token_id,
                generation_config,
            )
        tokens = output.text_tokens
        cots = []
        for i in range(len(tokens)):
            cot_out = vl_chat_processor.tokenizer.decode(tokens[i], skip_special_tokens=True)
            cots.append(cot_out)
        return cots, output.gen_img

    def inference_from_prompt(prompt):
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            # logits = logit_cond
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)


        dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        return visual_img
        
    jsonl_path = os.path.expanduser(jsonl_path)
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        if accelerator.num_processes > 1:
            ids = list(range(accelerator.process_index, len(lines)//accelerator.num_processes*accelerator.num_processes, accelerator.num_processes))
            lines = lines[accelerator.process_index:len(lines)//accelerator.num_processes*accelerator.num_processes:accelerator.num_processes]
        else:
            ids = list(range(len(lines)))

    for idx, line in tqdm(zip(ids, lines), total=len(lines)):
        data = json.loads(line)
        text = data["prompt"]
        prompt = get_prompt(text, cot=cot)
        this_out_dir = os.path.join(out_dir,f"{idx:05d}")
        os.makedirs(this_out_dir, exist_ok=True)
        meta_data_path = os.path.join(this_out_dir, "metadata.jsonl")
        sample_out_dir = os.path.join(this_out_dir, "samples")
        if os.path.exists(sample_out_dir):
            png_list = os.listdir(sample_out_dir)
            if len(png_list) == parallel_size:
                continue
        with open(os.path.join(meta_data_path), "w") as f:
            f.write(json.dumps(data))
        if cot:
            cots, visual_img=inference_from_prompt_cot(prompt)
            cot_out_dir = os.path.join(this_out_dir, "cots")
            os.makedirs(cot_out_dir, exist_ok=True)
            for i in range(len(cots)):
                with open(os.path.join(cot_out_dir, f"{i:04d}.txt"), "w") as f:
                    f.write(cots[i])
        else:
            visual_img=inference_from_prompt(prompt)
        os.makedirs(sample_out_dir, exist_ok=True)
        for i in range(parallel_size):
            save_path = os.path.join(sample_out_dir, f"{i:04d}.png")
            PIL.Image.fromarray(visual_img[i]).save(save_path)


if __name__ == "__main__":
    generate_from_geneval_jsonl(
        "~/project/geneval/prompts/evaluation_metadata.jsonl",
        vl_gpt,
        vl_chat_processor,
        # prompt,
        out_dir,
        cot=use_cot
    )