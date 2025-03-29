import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

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
        "content": f"Perform reasoning about how you would generate the following image.\n Image description: One apple and two bananas in a plate on a table.",
    },
    {"role": "<|Assistant|>", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )

# input_ids = vl_chat_processor.tokenizer.encode(sft_format)
# input_ids = torch.LongTensor(input_ids).to(vl_gpt.device)
# inputs_embeds = vl_chat_processor.language_model.get_input_embeddings()(input_ids)


# load images and prepare for inputs
# pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=[], force_batchify=True
).to(vl_gpt.device)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# # run the model to get the response
# outputs = vl_gpt.language_model.generate(
#     inputs_embeds=inputs_embeds,
#     attention_mask=prepare_inputs.attention_mask,
#     pad_token_id=tokenizer.eos_token_id,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     max_new_tokens=512,
#     do_sample=False,
#     use_cache=True,
# )

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    conversation: str,
    temperature: float = 1,
    max_tokens: int = 1024,
    do_sample: bool = False,
    use_cache: bool = False,
):
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=[], force_batchify=True
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    generated_tokens = torch.zeros((1, max_tokens), dtype=torch.int).cuda()

    for i in range(max_tokens):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=use_cache, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.language_model.lm_head(hidden_states[:, -1, :])
        if not do_sample:
            next_token = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        next_emb = mmgpt.language_model.get_input_embeddings()(next_token)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        inputs_embeds = next_emb.unsqueeze(dim=1)
    
    return generated_tokens

generated_tokens = generate(vl_gpt, vl_chat_processor, conversation, temperature=1, max_tokens=128, do_sample=False, use_cache=True)
for i in range(1):
    answer = tokenizer.decode(generated_tokens[i].cpu().tolist(), skip_special_tokens=True)
    print(f"{prepare_inputs['sft_format'][0]}", answer)