import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


# specify the path to the model
model_path = "/home/v-zhangyu3/expdata/janus_sft/sft_test/global_step_40"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained("deepseek-ai/Janus-Pro-7B")
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

template = "First reason in detail about how you can generate an image with the following prompt. Then generate the image. Prompt: {}"
prompt = "an ornamental snowflake on a gray background"
conversation = [
    {
        "role": "<|User|>",
        "content": template.format(prompt),
    },
    {"role": "<|Assistant|>", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)

prompt = sft_format 
input_ids = vl_chat_processor.tokenizer.encode(prompt)
input_ids = torch.LongTensor(input_ids).cuda()
attention_mask = torch.ones((1, len(input_ids)), dtype=torch.bool).cuda()
do_sample = True
max_new_tokens = 1024
eos_token_id = vl_chat_processor.tokenizer.eos_token_id
pad_token_id = vl_chat_processor.tokenizer.pad_token_id
image_start_token_id = vl_chat_processor.image_start_id

output = vl_gpt.text_img_generate(
    input_ids.unsqueeze(0),
    attention_mask, 
    do_sample, 
    max_new_tokens, 
    eos_token_id, 
    pad_token_id, 
    image_start_token_id,
)

cot_token = output.text_tokens
cot = vl_chat_processor.tokenizer.decode(cot_token[0], skip_special_tokens=False)
print()
print()
print(cot)
os.makedirs('generated_samples', exist_ok=True)
save_path = os.path.join('generated_samples', "img_cot.jpg")
PIL.Image.fromarray(output.gen_img[0]).save(save_path)
print("Images saved to 'generated_samples' folder")
