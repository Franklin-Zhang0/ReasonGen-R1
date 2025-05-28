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
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
import PIL
import numpy as np
from datasets import load_from_disk, concatenate_datasets, load_dataset
import io
import base64
import random


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 response_key='response',
                 response_dict_keys=None,
                 max_length=1024,
                 truncation='error'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = [] if not prompt_dict_keys else prompt_dict_keys
        self.response_dict_keys = [] if not response_dict_keys else response_dict_keys

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):

        def series_to_item(ls):
            import pandas, numpy
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.prompts={self.prompts}')
                raise
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.responses={self.responses}')
                raise
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{'role': 'user', 'content': prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors='pt', add_special_tokens=False)
        response_ids = response_ids_output['input_ids'][0]
        response_attention_mask = response_ids_output['attention_mask'][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }

def pad_to_square(image):
    """
    Pad the image to make it square.
    """
    w, h = image.size
    size = max(w, h)
    new_image = PIL.Image.new("RGB", (size, size), (255, 255, 255))
    new_image.paste(image, ((size - w) // 2, (size - h) // 2))
    return new_image

def random_crop_to_square(image):
    """
    Randomly crop the image to make it square.
    """
    w, h = image.size
    size = min(w, h)
    left = np.random.randint(0, w - size + 1)
    top = np.random.randint(0, h - size + 1)
    right = left + size
    bottom = top + size
    
    return image.crop((left, top, right, bottom))

def center_crop_to_square(image):
    """
    Center crop the image to make it square.
    """
    w, h = image.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    right = left + size
    bottom = top + size
    
    return image.crop((left, top, right, bottom))
        
def preprocess_img(image, size=(384, 384), processing=None):
    if isinstance(image, str): #base64 
        # data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQE ...
        image = image.split(";base64,")[1]
        image = io.BytesIO(base64.b64decode(image))
        image = PIL.Image.open(image).convert('RGB')
    
    #pad to square
    if processing == 'random_crop':
        image = random_crop_to_square(image)
    elif processing == 'center_crop':
        image = center_crop_to_square(image)
    elif processing == 'pad':
        image = pad_to_square(image)
    elif processing is None:
        pass
    else:
        raise NotImplementedError(f'Unknown processing method {processing}')
    
    image = image.resize(size)
    image = np.array(image)
    image = image.astype(np.float32)  # Convert to float32
    image = image / 255.0 * 2 - 1  # Normalize to [-1, 1]
    image = np.transpose(image, (2, 0, 1)) # Change to (C, H, W)
    return image  

dummy_reasoning = """
1. Identify the objects:
   - One apple
   - Two bananas
   - One plate

2. Determine their count and arrangement:
   - The apple and bananas should clearly be distinguishable as separate fruits.
   - The apple should not be duplicated — exactly one.
   - The bananas should be visible as two, and ideally curved slightly for realism.

3. Layout design:
   - The apple could be placed in the center or to one side of the plate.
   - The two bananas might be curved around the apple or placed parallel next to it.
   - The plate should be visible underneath all three fruits to fulfill the “on a plate” requirement.

4. Background and context:
   - A neutral or kitchen table background works best — nothing too busy.
   - Lighting should be soft to enhance the natural look of the fruits.
"""

class DummySFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 processor,
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 response_key='response',
                 response_dict_keys=None,
                 max_length=1024,
                 truncation='error'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

        self.max_length = max_length
        self.template = "First reason in detail about how you can generate an image with the following prompt. Then generate the image. Prompt: {}"
        self.prompt = "One apple and two bananas on a plate"
        self.cot = dummy_reasoning
        self.image = PIL.Image.new('RGB', (384, 384), color='white') 
        self.image_token_num_per_image = 576
        self.img_size = 384   

    def __len__(self):
        return 128

    def __getitem__(self, item):
        
        prompt = self.prompt
        response = self.cot
        # pil to np array
        image = preprocess_img(self.image)

        # apply chat template
        prompt_chat = [
            {'role': "<|User|>", 'content': self.template.format(prompt)},
            {'role': "<|Assistant|>", 'content': ""}
            ]

        # string
        prompt_chat_str = self.processor.apply_sft_template_for_multi_turn_prompts(prompt_chat, sft_format=self.processor.sft_format, system_prompt="")
        response_chat_str = response #+ self.processor.image_start_tag

        # tokenize
        prompt_ids_output = self.tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]

        response_ids_output = self.tokenizer(response_chat_str, return_tensors='pt', add_special_tokens=False)
        response_ids = response_ids_output['input_ids'][0]
        response_attention_mask = response_ids_output['attention_mask'][0]
        
        img_indices = [len(response_ids)]
        response_ids = self.processor.add_image_token(input_ids = response_ids, image_indices=img_indices)[0]
        response_attention_mask = torch.cat((response_attention_mask, torch.ones((len(response_ids) - len(response_attention_mask)))), dim=0)

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]
        
        one = torch.ones((1,), dtype=torch.long)
        true = torch.ones((1,), dtype=torch.bool)

        input_ids = torch.cat((self.tokenizer.bos_token_id * one, prompt_ids, response_ids, self.tokenizer.eos_token_id * one), dim=-1)
        attention_mask = torch.cat((true, prompt_attention_mask, response_attention_mask, true), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)
        img_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        img_mask[input_ids == self.processor.image_id] = True

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
            'input_img_mask': img_mask,
            'pixel_values': image
        }
        
class HFSFTDataset(Dataset):

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 processor,
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 response_key='response',
                 response_dict_keys=None,
                 image_key='image',
                 max_length=1024,
                 truncation='error',
                 template="",
                 prompt_augmentation: List[str]=None, # augment prompt key here
                 cot_augmentation: List[str]=None, # augment response key here
                 prompt_dropout: float=0.0,
                 two_stage: bool=False,
                 ):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation
        
        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]
            
        dataset_list = []
        for file in parquet_files:
            if '@' in file:
                path, split = file.split('@')
                dataset = self.load_dataset(path)
                if len(list(dataset.keys())) == 1:
                    key = list(dataset.keys())[0]
                    dataset = dataset[key].train_test_split(test_size=0.05, seed=42)[split]
                else:
                    dataset = dataset[split]
                if split == 'train':
                    self.image_processing = 'random_crop'
                else:
                    self.image_processing = 'center_crop'
            else:
                dataset = self.load_dataset(file)
                self.image_processing = 'random_crop'
            dataset_list.append(dataset)
            
        self.dataset = concatenate_datasets(dataset_list)

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.cot_key = response_key
        self.image_key = image_key

        self.max_length = max_length
        self.template = template
        self.image_token_num_per_image = 576
        self.img_size = 384   
        self.prompt_augmentation = prompt_augmentation
        self.cot_augmentation = cot_augmentation
        self.prompt_dropout = prompt_dropout
        self.two_stage = two_stage
        # self.dataset = self.dataset.filter(lambda x: x[self.cot_key] != "")
    
    def load_dataset(self, path):
        try:
            dataset = load_dataset(path)
        except:
            dataset = load_from_disk(path)
        return dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def pad_to_max_length(self, input_ids, attention_mask):
        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')
        
        return input_ids, attention_mask

    def __getitem__(self, item):
        data = self.dataset[item]
        prompt = data[self.prompt_key]
        response = data[self.cot_key]
        image = data[self.image_key]
        # pil to np array
        image = preprocess_img(image, processing=self.image_processing)
        
        if self.prompt_augmentation is not None:
            prompt = prompt_augmentation(data, self.prompt_augmentation)
        if self.cot_augmentation is not None:
            response = cot_augmentation(data, self.cot_augmentation)
        # make the first letter small letter
        # prompt = prompt[0].lower() + prompt[1:]
        # remove the last dot
        if prompt[-1] == '.':
            prompt = prompt[:-1]
        
        # apply chat template
        prompt_chat = [
            {'role': "<|User|>", 'content': self.template.format(prompt)},
            {'role': "<|Assistant|>", 'content': ""}
            ]
        # string
        prompt_chat_str = self.processor.apply_sft_template_for_multi_turn_prompts(prompt_chat, sft_format=self.processor.sft_format, system_prompt="" if not self.two_stage else "You are a helpful assistant. Given a short prompt, generate a richly detailed image description suitable for guiding an image generation model, including objects, setting, spatail relationship, mood, and visual details.")
        response_chat_str = response #+ self.processor.image_start_tag
        
        if self.two_stage:
            cot_as_prompt_chat = [
                {'role': "<|User|>", 'content': self.template.format(prompt)+response},
                {'role': "<|Assistant|>", 'content': ""}
                ]
            cot_as_prompt_chat_str = self.processor.apply_sft_template_for_multi_turn_prompts(cot_as_prompt_chat, sft_format=self.processor.sft_format, system_prompt="")
            cot_as_prompt_ids_output = self.tokenizer(cot_as_prompt_chat_str, return_tensors='pt', add_special_tokens=False)
            cot_as_prompt_ids = cot_as_prompt_ids_output['input_ids'][0]
            cot_as_prompt_attention_mask = cot_as_prompt_ids_output['attention_mask'][0]
        
        # tokenize
        prompt_ids_output = self.tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]

        response_ids_output = self.tokenizer(response_chat_str, return_tensors='pt', add_special_tokens=False)
        response_ids = response_ids_output['input_ids'][0]
        response_attention_mask = response_ids_output['attention_mask'][0]
        
        to_drop_prompt = random.random() < self.prompt_dropout
        
        if not self.two_stage:
            if to_drop_prompt: # replace prompt with padding, attention mask = 0
                prompt_ids = torch.ones_like(prompt_ids, dtype=torch.long) * self.tokenizer.pad_token_id
                prompt_attention_mask = torch.zeros_like(prompt_attention_mask, dtype=torch.bool)
                response_ids = torch.ones_like(response_ids, dtype=torch.long) * self.tokenizer.pad_token_id
                response_attention_mask = torch.zeros_like(response_attention_mask, dtype=torch.bool)
            
            img_indices = [len(response_ids)]
            
            response_ids = self.processor.add_image_token(input_ids = response_ids, image_indices=img_indices)[0]
            response_attention_mask = torch.cat((response_attention_mask, torch.ones((len(response_ids) - len(response_attention_mask)))), dim=0)
        
            # add bos and eos token
            prompt_length = prompt_ids.shape[0] + 1
            response_length = response_ids.shape[0] + 1
            
            one = torch.ones((1,), dtype=torch.long)
            true = torch.ones((1,), dtype=torch.bool)

            input_ids = torch.cat((self.tokenizer.bos_token_id * one, prompt_ids, response_ids, self.tokenizer.eos_token_id * one), dim=-1)
            attention_mask = torch.cat((true, prompt_attention_mask, response_attention_mask, true), dim=-1)
            
            input_ids, attention_mask = self.pad_to_max_length(input_ids, attention_mask)

            position_ids = compute_position_id_with_mask(attention_mask)
            img_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            img_mask[input_ids == self.processor.image_id] = True

            loss_mask = attention_mask.clone()
            if prompt_length > 1:
                # mask out prompt for SFT.
                loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
            # mask out the last token in response
            loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'loss_mask': loss_mask,
                'input_img_mask': img_mask,
                'pixel_values': image
            }
            
        else: # seperate text and image training
            # text inputs
            text_input_ids = torch.cat((self.tokenizer.bos_token_id * torch.ones((1,), dtype=torch.long), prompt_ids, response_ids, self.tokenizer.eos_token_id * torch.ones((1,), dtype=torch.long)), dim=-1)
            text_attention_mask = torch.cat((torch.ones((1,), dtype=torch.bool), prompt_attention_mask, response_attention_mask, torch.ones((1,), dtype=torch.bool)), dim=-1)
            text_input_ids, text_attention_mask = self.pad_to_max_length(text_input_ids, text_attention_mask)
            text_position_ids = compute_position_id_with_mask(text_attention_mask)
            text_loss_mask = text_attention_mask.clone()
            prompt_length = prompt_ids.shape[0] + 1
            response_length = response_ids.shape[0] + 1
            if prompt_length > 1:
                # mask out prompt for SFT.
                text_loss_mask[:min(prompt_length, text_loss_mask.size(0)) - 1] = 0
            text_loss_mask[min(prompt_length + response_length, text_loss_mask.size(0)) - 1] = 0
            
            # image inputs, treat the cot as prompt
            if to_drop_prompt: # replace prompt with padding, attention mask = 0
                cot_as_prompt_ids = torch.ones_like(cot_as_prompt_ids, dtype=torch.long) * self.tokenizer.pad_token_id
                cot_as_prompt_attention_mask = torch.zeros_like(cot_as_prompt_attention_mask, dtype=torch.bool)
            img_indices = [len(cot_as_prompt_ids)]
            cot_as_prompt_ids = self.processor.add_image_token(input_ids = cot_as_prompt_ids, image_indices=img_indices)[0]
            cot_as_prompt_attention_mask = torch.cat((cot_as_prompt_attention_mask, torch.ones((len(cot_as_prompt_ids) - len(cot_as_prompt_attention_mask)))), dim=0)
            
            img_input_ids = torch.cat((self.tokenizer.bos_token_id * torch.ones((1,), dtype=torch.long), cot_as_prompt_ids),  dim=-1)
            img_attention_mask = torch.cat((torch.ones((1,), dtype=torch.bool), cot_as_prompt_attention_mask), dim=-1)

            img_prompt_length = img_indices[0] + 1
            img_response_length = len(img_input_ids) - img_prompt_length + 1
            
            
            img_input_ids, img_attention_mask = self.pad_to_max_length(img_input_ids, img_attention_mask)
            img_position_ids = compute_position_id_with_mask(img_attention_mask)
            img_loss_mask = img_attention_mask.clone()
            img_input_img_mask = torch.zeros_like(img_attention_mask, dtype=torch.bool)
            img_input_img_mask[img_input_ids == self.processor.image_id] = True
            if prompt_length > 1:
                # mask out prompt for SFT.
                img_loss_mask[:min(img_prompt_length, img_loss_mask.size(0)) - 1] = 0
            img_loss_mask[min(img_prompt_length + img_response_length, img_loss_mask.size(0)) - 1] = 0
            
            return {
                'text_input_ids': text_input_ids,
                'text_attention_mask': text_attention_mask,
                'text_position_ids': text_position_ids,
                'text_loss_mask': text_loss_mask,
                'img_input_ids': img_input_ids,
                'img_attention_mask': img_attention_mask,
                'img_position_ids': img_position_ids,
                'img_loss_mask': img_loss_mask,
                'img_input_img_mask': img_input_img_mask,
                'pixel_values': image
            }
            
                
            
            
            
            
        
        
def prompt_augmentation(data, keys):
    num_augmentations = len(keys)
    selected_key_idx = random.randint(0, num_augmentations - 1)
    selected_key = keys[selected_key_idx]
    if selected_key == 'longIB_captions':
        return data[selected_key]
    elif selected_key == 'short_caption':
        return data['augmented_prompts'][selected_key]
    elif selected_key in ['paraphrases', 'varied_captions']:
        prompts = data['augmented_prompts'][selected_key]
        if type(prompts) == str: # legacy
            prompts = eval(prompts)
        selected_idx = random.randint(0, len(prompts) - 1)
        return prompts[selected_idx]
    elif selected_key == 'tags':
        shuffled_tags:list[str] = data['augmented_prompts'][selected_key]
        return ",".join(shuffled_tags)
    elif selected_key == 'object_prompts':
        objects = data['augmented_prompts'][selected_key]
        return ", ".join(objects[:-1]) + " and " + objects[-1]
    else:
        raise NotImplementedError(f'Unknown augmentation key {selected_key}')

def cot_augmentation(data, keys):
    num_augmentations = len(keys)
    selected_key_idx = random.randint(0, num_augmentations - 1)
    selected_key = keys[selected_key_idx]
    selected_key_tag = '<{}>'.format(selected_key) 
    '''
    detailed_caption,step_by_step,object_centric,tags,region_descriptions'''
    if selected_key == 'detailed_caption':
        return selected_key_tag + data[selected_key]
    elif selected_key == 'step_by_step':
        return selected_key_tag + data['augmented_cots'][selected_key]
    elif selected_key in ['object_centric', 'region_descriptions']:
        cots = data['augmented_cots'][selected_key]
        cot = "; ".join(cots)
        return selected_key_tag + cot
    elif selected_key == 'tags':
        return selected_key_tag + ", ".join(data['augmented_cots'][selected_key])

