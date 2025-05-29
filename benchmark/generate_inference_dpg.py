import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import json
from tqdm import tqdm
import tyro

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
from accelerate import Accelerator
accelerator = Accelerator()


# specify the path to the model

available_models={
    "Janus-Pro-7B": {
        "model_path": "deepseek-ai/Janus-Pro-7B", 
        "use_cot": False,
        "template": "A photo of {}."
    },
    "ReasonGen-R1":{
        "model_path":"Franklin0/ReasonGen-R1",
        "use_cot": True,
        "template": "A photo of {}. Output a richly detailed prompt: "
    },
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
out_dir = os.path.expanduser(f"~/project/Image-RL/dpg_benchmark/dpg_result/{model_name}/generated_images")
os.makedirs(out_dir, exist_ok=True)
model_path = available_models[model_name]["model_path"]
use_cot = available_models[model_name]["use_cot"]
use_two_stage = "two_stage" in available_models[model_name]

processor_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(processor_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).eval().to(accelerator.device)


if "template" in available_models[model_name]:
    template = available_models[model_name]["template"]
else:
    template = "A photo of {}. A richly detailed prompt: "


def get_prompt(text, cot = False):
    text = text.replace("A photo of", "").replace("a photo of", "").strip() # avoid redundant a photo of
    conversation = [
        {
            "role": "<|User|>",
            "content": template.format(text),
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
def generate_from_dpg_folder(
    dpg_prompt_dir: str,
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    out_dir: str,
    temperature: float = 1,
    parallel_size: int = 4,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    cot: bool = False,
):  
    def inference_from_prompt_cot(prompt,two_stage_img_sft_format=None):
        if two_stage_img_sft_format is not None:
            img_input_ids = vl_chat_processor.tokenizer.encode(two_stage_img_sft_format)
            img_input_ids = torch.LongTensor(img_input_ids).cuda()
            img_input_ids = img_input_ids.unsqueeze(0).repeat(parallel_size, 1)
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).cuda()
        attention_mask = torch.ones((len(input_ids)), dtype=torch.bool).cuda()
        do_sample = True
        max_new_tokens = 1024
        eos_token_id = vl_chat_processor.tokenizer.eos_token_id
        pad_token_id = vl_chat_processor.tokenizer.pad_token_id
        image_start_token_id = vl_chat_processor.image_start_id
        input_ids = input_ids.unsqueeze(0).repeat(parallel_size, 1)
        attention_mask = attention_mask.unsqueeze(0).repeat(parallel_size, 1)
        generation_config = {'cfg_weight': 5.0}
        if two_stage_img_sft_format is not None:
            output = vl_gpt.text_img_generate_two_stage(
                input_ids,
                img_input_ids,
                attention_mask, 
                do_sample, 
                max_new_tokens, 
                eos_token_id, 
                pad_token_id, 
                image_start_token_id,
                generation_config,
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
   
    dpg_prompt_dir = os.path.expanduser(dpg_prompt_dir)
    txt_files = os.listdir(dpg_prompt_dir)
    if accelerator.num_processes > 1:
        ids = list(range(accelerator.process_index, len(txt_files)//accelerator.num_processes*accelerator.num_processes, accelerator.num_processes))
        txt_files = txt_files[accelerator.process_index:len(txt_files)//accelerator.num_processes*accelerator.num_processes:accelerator.num_processes]
    else:
        ids = list(range(len(txt_files)))
    for idx, this_txt_file in tqdm(zip(ids, txt_files), total=len(txt_files)):
        with open(os.path.join(dpg_prompt_dir, this_txt_file), "r") as f:
            text = f.read()
        assert text is not None and isinstance(text, str)

        prompt = get_prompt(text, cot=cot)
        img_sft_format = None
        this_img_path = os.path.join(out_dir, this_txt_file.replace(".txt", ".png"))
        if os.path.exists(this_img_path):
            continue
        if cot:
            cots, visual_img=inference_from_prompt_cot(prompt, img_sft_format)
        else:
            visual_img=inference_from_prompt(prompt)

        # 4 images to 2*2 image
        visual_img_2_2 = np.zeros((img_size*2, img_size*2, 3), dtype=visual_img.dtype)
        visual_img_2_2[:img_size, :img_size, :] = visual_img[0]
        visual_img_2_2[:img_size, img_size:, :] = visual_img[1]
        visual_img_2_2[img_size:, :img_size, :] = visual_img[2]
        visual_img_2_2[img_size:, img_size:, :] = visual_img[3]
        assert img_size == 384
        PIL.Image.fromarray(visual_img_2_2).save(this_img_path)


if __name__ == "__main__":
    generate_from_dpg_folder(
        "~/project/ELLA/dpg_bench/prompts",
        vl_gpt,
        vl_chat_processor,
        out_dir,
        cot=use_cot
    )