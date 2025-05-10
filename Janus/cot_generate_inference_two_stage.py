import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


# specify the path to the model
# model_path = "/blob/franklin/ckpt/image_rl/janus_sft/200k_sample_aug_long/200k_sample_aug_long_7B_bs128_lr1e-5_image_1.0_text_0.5-free_template-0509/global_step_1650"
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained("deepseek-ai/Janus-Pro-7B")
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

template = "{}. "
# prompt = "A white bowl filled with halved, seasoned red potatoes is centrally positioned on a red surface. Surrounding it are smaller white bowls containing various ingredients: crispy bacon pieces, chopped radishes, diced avocado, sliced green onions on a blue plate, yellow corn kernels, fennel seeds, mustard seeds, fresh dill, and sliced green chili peppers. A silver spoon rests inside the potato bowl. No humans are visible in the image."
# prompt = "A person with light brown hair styled in loose waves is wearing a delicate golden headband adorned with small white pearls and intricate leaf designs. The background is blurred greenery, suggesting an outdoor setting. The person is dressed in a white garment with thin straps."
# prompt = 'A single light blue vintage scooter with a white seat and a transparent windshield is positioned in profile facing left against a yellow and white horizontally striped background. The scooter has white-walled tires and chrome details. Text overlays the image vertically and horizontally, with \"Polish Funk\" prominently displayed in large white and orange stylized font at the bottom. No humans are present in the image.'
# prompt = "An apple and two bananas in a plate on a table. The apple is red and the bananas are yellow. The plate is white and round. The table is brown and made of wood. The background is blurred."
# prompt = "A shiny blue vintage car with chrome detailing and silver rims is parked on a paved driveway in front of green trees and houses."
# prompt = 'a bicycle with a black seat and has a basket on the front on a street with a red brick wall and a white fence in the background'
prompt = 'an apple two bananas in a plate on a table'

this_id = "test"

conversation = [
    {
        "role": "<|User|>",
        "content": template.format(prompt),
    },
    {"role": "<|Assistant|>", "content": ""},
]
system_prompt = "You are a helpful assistant. Given a short prompt, generate a richly detailed image description suitable for guiding an image generation model, including objects, setting, spatail relationship, mood, and visual details."
sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt=system_prompt,
)

img_sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt=""
)

prompt = sft_format 
input_ids = vl_chat_processor.tokenizer.encode(prompt)
input_ids = torch.LongTensor(input_ids).cuda()
attention_mask = torch.ones((len(input_ids)), dtype=torch.bool).cuda()
img_input_ids = vl_chat_processor.tokenizer.encode(img_sft_format)
img_input_ids = torch.LongTensor(img_input_ids).cuda()
do_sample = True
max_new_tokens = 1024
eos_token_id = vl_chat_processor.tokenizer.eos_token_id
pad_token_id = vl_chat_processor.tokenizer.pad_token_id
image_start_token_id = vl_chat_processor.image_start_id
input_ids = input_ids.unsqueeze(0).repeat(4, 1)
img_input_ids = img_input_ids.unsqueeze(0).repeat(4, 1)
attention_mask = attention_mask.unsqueeze(0).repeat(4, 1)
generation_config = {'cfg_weight': 5.0}
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

tokens = output.text_tokens
model_name = model_path.split("/")[-2]
path = os.path.join("generated_samples", model_name, f'cfg{generation_config['cfg_weight']}', 'two_stage')
os.makedirs(path, exist_ok=True)
for i in range(len(tokens)):
    cot = vl_chat_processor.tokenizer.decode(tokens[i], skip_special_tokens=True)
    print()
    print()
    print(cot)
    with open(f"{path}/cot_{i}.txt", "w") as f:
        f.write(cot)
    print(f"Text saved to '{path}/cot_{i}.txt'")
    save_path = os.path.join(f'{path}', "img_cot_{}.png".format(i))
    PIL.Image.fromarray(output.gen_img[i]).save(save_path)
    print(f"Images saved to '{save_path}' folder")
