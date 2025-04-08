# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch

import os
import PIL
import datetime

class ImageGenerationRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, eval=False, img_saving_args={}) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.steps = 0
        self.save_freq = img_saving_args.save_freq
        self.save_num = img_saving_args.num
        self.save_path = img_saving_args.path
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_path = os.path.join(self.save_path, time_stamp)
        if eval:
            self.save_path = os.path.join(self.save_path, "eval")
        else:
            self.save_path = os.path.join(self.save_path, "train")
        
    def save_img(self, data: DataProto):
        gen_img = data.batch['gen_img']
        gen_img = gen_img.to('cpu').numpy() if isinstance(gen_img, torch.Tensor) else gen_img
        step_dir = os.path.join(self.save_path, str(self.steps))
        os.makedirs(step_dir, exist_ok=True)
        
        for i in range(min(len(gen_img), self.save_num)):
            save_path = os.path.join(step_dir, "img_{}.jpg".format(i))
            PIL.Image.fromarray(gen_img[i]).save(save_path)
            prompt = data.batch['prompts'][i]
            with open(os.path.join(step_dir, "prompts.txt".format(i)), 'a') as f:
                f.write(self.tokenizer.decode(prompt, skip_special_tokens=True))
                

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # save generated images
        if self.steps % self.save_freq == 0:
            self.save_img(data)
        self.steps += 1
            
        print("Images saved to 'generated_samples' folder")
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            reward_tensor[i, valid_response_length - 1] = torch.randint(0, 2, (1,)).float()
        
        
        return reward_tensor
        # already_print_data_sources = {}

        # for i in range(len(data)):
        #     data_item = data[i]  # DataProtoItem

        #     prompt_ids = data_item.batch['prompts']

        #     prompt_length = prompt_ids.shape[-1]

        #     valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        #     valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        #     response_ids = data_item.batch['responses']
        #     valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        #     valid_response_ids = response_ids[:valid_response_length]

        #     # decode
        #     prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        #     response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        #     ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

        #     data_source = data_item.non_tensor_batch['data_source']

        #     extra_info = data_item.non_tensor_batch.get('extra_info', None)

        #     score = self.compute_score(
        #         data_source=data_source,
        #         solution_str=response_str,
        #         ground_truth=ground_truth,
        #         extra_info=extra_info,
        #     )
        #     reward_tensor[i, valid_response_length - 1] = score

        #     if data_source not in already_print_data_sources:
        #         already_print_data_sources[data_source] = 0

        #     if already_print_data_sources[data_source] < self.num_examine:
        #         already_print_data_sources[data_source] += 1
        #         print("[prompt]", prompt_str)
        #         print("[response]", response_str)
        #         print("[ground_truth]", ground_truth)
        #         print("[score]", score)

        # return reward_tensor
