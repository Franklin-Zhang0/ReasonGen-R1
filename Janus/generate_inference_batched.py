import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import verl.utils.torch_functional as verl_F


# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "<|User|>",
        "content": "One apple and two bananas in a plate on a table.",
    },
    {"role": "<|Assistant|>", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format# + vl_chat_processor.image_start_tag

def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 2,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    # input_ids = vl_chat_processor.tokenizer.encode(prompt)
    # input_ids = torch.LongTensor(input_ids)
    # length = len(input_ids)
    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt,
                                                                         tokenizer=vl_chat_processor.tokenizer,
                                                                         max_length=256,
                                                                         pad_token_id=vl_chat_processor.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation='error')
    input_ids = input_ids[0]
    sentence_start_token, image_start_token = vl_chat_processor.tokenizer.encode(vl_chat_processor.image_start_tag)
    input_ids = torch.cat([torch.LongTensor([vl_chat_processor.tokenizer.pad_token_id]), input_ids, torch.LongTensor([image_start_token])])
    num_pad = torch.sum(input_ids == vl_chat_processor.tokenizer.pad_token_id, dim=-1)
    last_pad_idx = num_pad - 1
    input_ids[last_pad_idx] = sentence_start_token
    attention_mask = attention_mask[0]
    attention_mask = torch.cat([torch.LongTensor([0]), attention_mask, torch.LongTensor([1])])
    attention_mask[last_pad_idx] = 1
    attention_mask = attention_mask.unsqueeze(dim=0).repeat(parallel_size*2, 1)
    position_ids = compute_position_id_with_mask(attention_mask).cuda()
    # breakpoint()

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, :-1] = vl_chat_processor.pad_id
            tokens[i, last_pad_idx] = sentence_start_token

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    
    # breakpoint()
    cache_position = torch.range(0, len(input_ids)-1).cuda()
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, 
                                             use_cache=True, 
                                             past_key_values=outputs.past_key_values if i != 0 else None, 
                                             attention_mask=attention_mask,
                                             position_ids=position_ids,
                                             cache_position=cache_position)
        # outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, attention_mask=attention_mask, position_ids=position_ids)
        
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        
        # cached inputs
        inputs_embeds = img_embeds.unsqueeze(dim=1)
        position_ids = (position_ids[:,-1:] + 1).cuda()
        attention_mask = torch.cat([attention_mask, torch.ones((parallel_size*2, 1), dtype=torch.int)], dim=1)
        
        # breakpoint()
        cache_position = cache_position[-1:] + 1
        
        # inputs_embeds = torch.cat([inputs_embeds, img_embeds.unsqueeze(dim=1)], dim=1)
        # attention_mask = torch.cat([attention_mask, torch.ones((parallel_size*2, 1), dtype=torch.int)], dim=1)
        # position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=-1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)
    print("Images saved to 'generated_samples' folder")

import time
start = time.time()
generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
)
print("Time taken: ", time.time()-start)