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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import logging
import re
from contextlib import nullcontext
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.dataset.sft_dataset import SFTDataset, DummySFTDataset, HFSFTDataset
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from peft import LoraConfig, TaskType, get_peft_model

from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl import DataProto
from janus.models import MultiModalityCausalLM, VLChatProcessor
from copy import deepcopy
import torch.nn.functional as F
from verl.utils.torch_functional import logprobs_from_logits
from verl.trainer.ppo.core_algos import kl_penalty
import numpy as np

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))


def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


class FSDPSFTTrainer(object):

    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        # build tokenizer first
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
        if 'Janus' in local_model_path: # janus is not in huggingface cfg yet
            self.processor = VLChatProcessor.from_pretrained(local_model_path)
            self.tokenizer = self.processor.tokenizer
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.image_start_token_id = self.processor.image_start_id
            self.image_end_token_id = self.processor.image_end_id
            self.bos_token_id = self.tokenizer.bos_token
        else:
            from verl.utils import hf_tokenizer
            self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
            self.processor = None
            if self.config.data.chat_template is not None:
                raise ValueError('Apply Chat template from config is not supported yet.')

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, 'ulysses_sequence_parallel_size', 1)
        self.use_remove_padding = getattr(self.config, 'use_remove_padding', False)
        if self.device_mesh.get_rank() == 0:
            print(f'Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}')
            print(f'Using remove padding: {self.use_remove_padding}')
        
        self._build_dataloader()
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f'Normalize batch size by dp {dp_size}')

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self):
        config = self.config
        # build dataset
        self.train_dataset = HFSFTDataset(parquet_files=config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        processor=self.processor,
                                        prompt_key=config.data.prompt_key,
                                        prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                        response_key=config.data.response_key,
                                        response_dict_keys=config.data.get('response_dict_keys', None),
                                        max_length=config.data.max_length,
                                        truncation=config.data.truncation,
                                        template=config.data.chat_template,
                                        prompt_augmentation=config.data.get('prompt_augmentation', None),
                                        cot_augmentation= config.data.get('cot_augmentation', None),
                                        prompt_dropout=config.data.get('prompt_dropout', 0.0),
                                        two_stage=config.algorithm.get('two_stage', False),
                                        )
        self.val_dataset = HFSFTDataset(parquet_files=config.data.val_files,
                                      tokenizer=self.tokenizer,
                                      processor=self.processor,
                                      prompt_key=config.data.prompt_key,
                                      prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                      response_key=config.data.response_key,
                                      response_dict_keys=config.data.get('response_dict_keys', None),
                                      max_length=config.data.max_length,
                                      truncation=config.data.truncation,
                                      template=config.data.chat_template,
                                      prompt_augmentation=config.data.get('prompt_augmentation', None),
                                      cot_augmentation= config.data.get('cot_augmentation', None),
                                      two_stage=config.algorithm.get('two_stage', False)
                                      )

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank('dp')
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f'Using SP rank {rank} and size {world_size} for data distribution')
                print(f'Each SP rank gets different data, but the same data WITHIN the same rank')
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Using FSDP rank {rank} and size {world_size} for data distribution')

        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.data.train_batch_size,
                                           sampler=self.train_sampler,
                                           num_workers=8,
                                           pin_memory=True,
                                           drop_last=True)

        self.val_sampler = DistributedSampler(self.val_dataset,
                                              shuffle=False,
                                              num_replicas=world_size,
                                              rank=rank,
                                              drop_last=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=config.data.micro_batch_size_per_gpu,
                                         sampler=self.val_sampler,
                                         num_workers=8,
                                         pin_memory=True,
                                         drop_last=True)

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage('Before model allocation', logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context():
            if 'Janus' in local_model_path: # janus is not in huggingface cfg yet
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_model_path,
                                                                torch_dtype=torch.bfloat16,
                                                                trust_remote_code=trust_remote_code)
                self.model.gen_vision_model.requires_grad = False
                self.vq_mudule = deepcopy(self.model.gen_vision_model)
                self.vq_mudule.eval()
            else:
                from verl.utils import hf_tokenizer, hf_processor
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float32,
                    attn_implementation='flash_attention_2',
                    trust_remote_code=True
                    )
                self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=trust_remote_code)
                self.processor = hf_processor(local_model_path, trust_remote_code=trust_remote_code)
            
            if self.config.algorithm.get('use_kl_loss', False):
                self.ref_model = deepcopy(self.model)
                self.ref_model.eval()
                for name, param in self.ref_model.named_parameters():
                # param here is each FSDP “flat” shard on this rank
                    param.requires_grad = False

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=self.model)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get('use_liger', False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get('lora_rank', 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none"
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        log_gpu_memory_usage('After model allocation', logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(self.model,
                                                config=self.config.model.fsdp_config.wrap_policy,
                                                is_lora=self.config.model.get('lora_rank', 0) > 0)
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(module=self.model,
                               auto_wrap_policy=auto_wrap_policy,
                               param_init_fn=init_fn,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               device_mesh=self.device_mesh,
                               sync_module_states=True,
                               device_id=torch.cuda.current_device(),
                               cpu_offload=cpu_offload,
                               use_orig_params=False)
        
        if self.config.algorithm.get('use_kl_loss', False):
            self.ref_fsdp_model = FSDP(module=self.ref_model,
                                       auto_wrap_policy=auto_wrap_policy,
                                       param_init_fn=init_fn,
                                       sharding_strategy=ShardingStrategy.FULL_SHARD,
                                       mixed_precision=mixed_precision,
                                       device_mesh=self.device_mesh,
                                       sync_module_states=True,
                                       device_id=torch.cuda.current_device(),
                                       cpu_offload=cpu_offload,
                                       use_orig_params=False)
            

        log_gpu_memory_usage('After FSDP wrapping', logger=logger)

        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                     lr=self.config.optim.lr,
                                     betas=self.config.optim.betas,
                                     weight_decay=self.config.optim.weight_decay)

        log_gpu_memory_usage('After initialize optimizer', logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f'Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}'
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=self.total_steps)
        
    def apply_loss_scale(self, loss, mask_dict):
        bs = mask_dict['image'].shape[0]
        loss = loss.reshape(bs, -1)
        loss_scale = self.config.algorithm.loss_scale
        gradual_increase_key = self.config.algorithm.loss_scale.gradual_increase_key
        start_ratio, end_ratio = self.config.algorithm.loss_scale.gradual_increase_interval
        start_ratio, end_ratio = float(start_ratio), float(end_ratio)
        
        loss_dict = {}
        for key in mask_dict.keys():
            loss_dict[key] = loss[mask_dict[key]]
            loss_dict[key] = torch.sum(loss_dict[key]) / (torch.sum(loss_dict[key]>0.0) + 1e-8)
            loss[mask_dict[key]] = loss[mask_dict[key]] * loss_scale[key]
            if key in gradual_increase_key:
                current_ratio = self.global_step / self.total_steps
                scale = (np.clip(current_ratio, start_ratio, end_ratio) - start_ratio) / (end_ratio - start_ratio)
                loss[mask_dict[key]] *= scale
        loss = loss.reshape(-1)
        return loss, loss_dict

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        device = torch.device(f"cuda:{self.device_mesh.get_rank()}")
        # Move inputs to GPU and prepare loss mask
        if not self.config.algorithm.get('two_stage', False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_ids = batch['position_ids'].to(device)
            input_img_mask = batch['input_img_mask'].to(device)
            loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).to(device)
            pixel_values = batch.pop('pixel_values').to(device)
        else:
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            text_position_ids = batch['text_position_ids'].to(device)
            text_loss_mask = batch.pop('text_loss_mask')[:, :-1].reshape(-1).to(device)
            img_input_ids = batch['img_input_ids'].to(device)
            img_attention_mask = batch['img_attention_mask'].to(device)
            img_position_ids = batch['img_position_ids'].to(device)
            img_loss_mask = batch.pop('img_loss_mask')[:, :-1].reshape(-1).to(device)
            img_input_img_mask = batch['img_input_img_mask'].to(device)
            pixel_values = batch.pop('pixel_values').to(device)
            
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                self.vq_mudule = self.vq_mudule.to(device)
                img_code = self.vq_mudule.encode(pixel_values)[2][2]
                        
            if self.config.algorithm.get('two_stage', False):
                img_input_ids[img_input_img_mask] = img_code
            else:
                input_ids[input_img_mask] = img_code
        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if not use_sp:
                    # Standard forward pass without sequence parallel
                    if not self.config.algorithm.get('two_stage', False):
                        labels = input_ids[:, 1:].contiguous()
                        img_mask = input_img_mask[:, 1:].contiguous()
                        img_start_token_mask = labels == self.image_start_token_id
                        text_mask = ~img_mask & ~img_start_token_mask
                        mask_dict = {
                            'image': img_mask,
                            'text': text_mask,
                            'image_start_token': img_start_token_mask
                        }
                        output = self.fsdp_model(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                position_ids=position_ids,
                                                input_img_mask=input_img_mask,
                                                bos_token_id=self.bos_token_id,
                                                pad_token_id=self.pad_token_id,
                                                image_start_token_id=self.image_start_token_id,
                                                cfg_weight=1.0,
                                                use_cache=False)
                        logits = output['logits'] if isinstance(output, dict) else output.logits

                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels.contiguous()
                        # Flatten the tokens
                        shift_logits = shift_logits.view(shift_logits.size(0) * shift_logits.size(1), shift_logits.size(2))
                        shift_labels = shift_labels.view(-1)
                        # Enable model parallelism
                        shift_labels = shift_labels.to(shift_logits.device)
                        loss = loss_fct(shift_logits, shift_labels)
                        loss = loss * loss_mask.to(loss.device)
                        loss, loss_dict = self.apply_loss_scale(loss, mask_dict)
                    else:
                        loss_scale = self.config.algorithm.loss_scale
                        
                        # two‐stage: first compute text loss
                        text_labels = text_input_ids[:, 1:].contiguous()
                        text_out = self.fsdp_model(
                            input_ids=text_input_ids,
                            attention_mask=text_attention_mask,
                            position_ids=text_position_ids,
                            input_img_mask=None,
                            use_cache=False
                        )
                        text_logits = text_out['logits'] if isinstance(text_out, dict) else text_out.logits
                        shift_text_logits = text_logits[..., :-1, :].contiguous().view(-1, text_logits.size(-1))
                        shift_text_labels = text_labels.view(-1).to(shift_text_logits.device)
                        text_loss = loss_fct(shift_text_logits, shift_text_labels)
                        text_loss = text_loss * text_loss_mask.to(text_loss.device)
                        text_loss = torch.sum(text_loss) / (torch.sum(text_loss_mask) + 1e-8)
                        
                        text_loss_for_backward = text_loss * loss_scale['text']
                        if do_backward:
                            text_loss_for_backward.backward()

                        # then compute image loss
                        img_labels = img_input_ids[:, 1:].contiguous()
                        img_out = self.fsdp_model(
                            input_ids=img_input_ids,
                            attention_mask=img_attention_mask,
                            position_ids=img_position_ids,
                            input_img_mask=img_input_img_mask,
                            bos_token_id=self.bos_token_id,
                            pad_token_id=self.pad_token_id,
                            image_start_token_id=self.image_start_token_id,
                            cfg_weight=1.0,
                            use_cache=False
                        )
                        img_logits = img_out['logits'] if isinstance(img_out, dict) else img_out.logits
                        shift_img_logits = img_logits[..., :-1, :].contiguous().view(-1, img_logits.size(-1))
                        shift_img_labels = img_labels.view(-1).to(shift_img_logits.device)
                        img_loss = loss_fct(shift_img_logits, shift_img_labels)
                        
                        img_start_token_mask = shift_img_labels == self.image_start_token_id
                        img_start_token_loss = img_loss * img_start_token_mask
                        img_mask = img_input_img_mask[:, 1:].contiguous().view(-1)
                        img_loss = img_loss * img_mask.to(img_loss.device)
                        
                        img_loss = torch.sum(img_loss) / (torch.sum(img_mask) + 1e-8)
                        img_start_token_loss = torch.sum(img_start_token_loss) / (torch.sum(img_start_token_mask) + 1e-8)
                        
                        img_loss_for_backward = img_loss * loss_scale['image'] + img_start_token_loss * loss_scale['image_start_token']
                        if do_backward:
                            img_loss_for_backward.backward()
                        

                        # combine
                        loss_dict = {'text': text_loss, 'image': img_loss, 'image_start_token': img_start_token_loss}
                        loss = text_loss * loss_scale['text'] + img_loss * loss_scale['image'] + img_start_token_loss * loss_scale['image_start_token']
                        # dummy loss mask for later valid_token_this_rank computation
                        loss_mask = torch.ones(1).to(text_loss.device)
                        
                else:
                    # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                    # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                    # 1. All SP ranks will receive the *SAME* batch
                    # 2. Different SP groups will receive *DIFFERENT* batches
                    # This is implemented by the DistributedSampler

                    batch_size, seqlen = input_ids.shape
                    # Remove padding
                    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                               attention_mask)  # input_ids_rmpad (total_nnz, ...)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                    # Unpad position_ids to align rotary
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                    # Pad and slice inputs for sequence parallelism
                    input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                    # For computing loss
                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                    # Forward pass
                    output = self.fsdp_model(
                        input_ids=input_ids_rmpad_sliced,
                        attention_mask=None,  # Not needed with flash attention varlen
                        position_ids=position_ids_rmpad_padded,
                        use_cache=False)

                    # Compute loss locally then aggregate
                    logits_rmpad = output.logits.squeeze(0)
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                    loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                    # Gather and unpad for sequence parallelism
                    loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                    # This is the loss collected from all ulysses ranks
                    full_loss = pad_input(hidden_states=loss.unsqueeze(-1),
                                          indices=indices,
                                          batch=batch_size,
                                          seqlen=seqlen)
                    full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                    full_loss = full_loss.reshape(-1)
                    loss_mask = loss_mask.to(full_loss.device)
                    loss = full_loss * loss_mask

                valid_token_this_rank = torch.sum(loss_mask)

                if self.config.data.balance_dp_token:
                    torch.distributed.all_reduce(valid_token_this_rank)
                    dp_size = self.ulysses_device_mesh.size('dp') if use_sp else torch.distributed.get_world_size()
                else:
                    dp_size = 1

                loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size
                
                if self.config.algorithm.use_kl_loss and not self.config.algorithm.get('two_stage', False):
                    with torch.no_grad():
                        self.ref_fsdp_model.eval()
                        ref_output = self.ref_fsdp_model(input_ids=input_ids,
                                                            attention_mask=attention_mask,
                                                            position_ids=position_ids,
                                                            input_img_mask=input_img_mask,
                                                            bos_token_id=self.bos_token_id,
                                                            pad_token_id=self.pad_token_id,
                                                            image_start_token_id=self.image_start_token_id,
                                                            cfg_weight=1.0,
                                                            use_cache=False)
                        ref_logits = ref_output['logits'] if isinstance(ref_output, dict) else ref_output.logits
                        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
                        shift_ref_logits = shift_ref_logits.view(shift_ref_logits.size(0) * shift_ref_logits.size(1),
                                                                    shift_ref_logits.size(2))
                        shift_ref_logits = shift_ref_logits.to(loss.device)
                    ref_log_prob = logprobs_from_logits(shift_ref_logits, shift_labels)
                    log_prob = logprobs_from_logits(shift_logits, shift_labels)
                    kl_loss = kl_penalty(log_prob, ref_log_prob, kl_penalty=self.config.algorithm.kl_penalty)
                    kl_loss = kl_loss * loss_mask.to(kl_loss.device)
                    kl_loss = torch.sum(kl_loss) / (valid_token_this_rank + 1e-8) * dp_size
                            
                    loss = loss + kl_loss * self.config.algorithm.kl_loss_weight
                
                if do_backward and not self.config.algorithm.get('two_stage', False):
                    loss.backward()
                return {
                    'loss': loss,
                    'kl_loss': kl_loss if self.config.algorithm.use_kl_loss else None,
                    'loss_dict': loss_dict
                }

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        step_kl_loss = 0
        loss_dict = {}
        for micro_batch in micro_batches:
            # loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            output = self._compute_loss_and_backward(batch=micro_batch)
            loss = output['loss'] / n_micro_batches
            if self.config.algorithm.use_kl_loss:
                kl_loss = output['kl_loss'] / n_micro_batches
                step_kl_loss += kl_loss.item()
            step_loss += loss.item()
            for key in output['loss_dict'].keys():
                if key not in loss_dict:
                    loss_dict[key] = 0
                loss_dict[key] += output['loss_dict'][key].item() / n_micro_batches

        self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage('Before optimizer step', logger=logger)

        self.optimizer.step()

        log_gpu_memory_usage('After optimizer step', logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage('After offload weights', logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        if self.config.algorithm.use_kl_loss:
            step_kl_loss = torch.tensor(step_kl_loss).cuda()
            torch.distributed.all_reduce(step_kl_loss, op=torch.distributed.ReduceOp.AVG)
        for key in loss_dict.keys():
            loss_dict[key] = torch.tensor(loss_dict[key]).cuda()
            torch.distributed.all_reduce(loss_dict[key], op=torch.distributed.ReduceOp.AVG)
            
        log_dict = {
            'train/loss'     : step_loss.detach().item(),
            'train/lr(1e-3)' : lr * 1e3,
        }
        for key in loss_dict.keys():
            log_dict[f'train/{key}'] = loss_dict[key].detach().item()
        if self.config.algorithm.use_kl_loss:
            log_dict['train/kl_loss'] = step_kl_loss.detach().item()
        return log_dict

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        seperate_loss = {}
        with torch.no_grad():
            output = self._compute_loss_and_backward(batch, do_backward=False)
            loss = output['loss']
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            if self.config.algorithm.use_kl_loss:
                kl_loss = output['kl_loss']
                torch.distributed.all_reduce(kl_loss, op=torch.distributed.ReduceOp.AVG)
            for key in output['loss_dict'].keys():
                seperate_loss[key] = output['loss_dict'][key]
                torch.distributed.all_reduce(seperate_loss[key], op=torch.distributed.ReduceOp.AVG)
        log_dict = {
            'val/loss': loss,
        }
        for key in output['loss_dict'].keys():
            log_dict[f'val/{key}'] = seperate_loss[key]
        if self.config.algorithm.use_kl_loss:
            log_dict['val/kl_loss'] = kl_loss
        return log_dict

    def save_checkpoint(self, step):
        # save checkpoint
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()

        path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
        # save huggingface model
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            self.processor.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger)

        self.global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        # TODO (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.
        # validation
        # val_logs = []
        # for data in self.val_dataloader:
        #     data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
        #     val_log = self.validation_step(data)
        #     val_logs.append(val_log)
        # if rank == 0:
        #     for key in val_logs[0].keys():
        #         val_metric = torch.mean(torch.stack([log[key] for log in val_logs]))
        #         metric = {key: val_metric.detach().item()}
        #         tracking.log(data=metric, step=self.global_step)
        # torch.distributed.barrier()
        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(self.train_dataloader,
                             total=self.steps_per_epoch,
                             desc=f"Epoch {epoch+1}/{self.config.trainer.total_epochs}"):
                self.global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=self.global_step)

                # for early exit validation
                if self.global_step >= self.total_training_steps or self.global_step % self.config.trainer.save_freq == 0:
                    # Save final checkpoint
                    self.save_checkpoint(step=self.global_step)
                    # Perform final validation
                    val_logs = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
                        val_log = self.validation_step(val_data)
                        val_logs.append(val_log)
                    if rank == 0:
                        for key in val_logs[0].keys():
                            val_metric = torch.mean(torch.stack([log[key] for log in val_logs]))
                            metric = {key: val_metric.detach().item()}
                            tracking.log(data=metric, step=self.global_step)
                    torch.distributed.barrier()
                    
                    if self.global_step >= self.total_training_steps:
                        return
                
            # save checkpoint
            self.save_checkpoint(step=self.global_step)
            # validation
            val_logs = []
            for data in self.val_dataloader:
                data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
                val_log = self.validation_step(data)
                val_logs.append(val_log)
            if rank == 0:
                for key in val_logs[0].keys():
                    val_metric = torch.mean(torch.stack([log[key] for log in val_logs]))
                    metric = {key: val_metric.detach().item()}
                    tracking.log(data=metric, step=self.global_step)
            torch.distributed.barrier()

            


# from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
import hydra

from torch.distributed.device_mesh import init_device_mesh

from verl.utils.distributed import initialize_global_process_group


@hydra.main(config_path='config', config_name='janus_sft_trainer', version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
                                           mesh_dim_names=('dp', 'sp'))
    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)
    trainer.fit()


if __name__ == '__main__':
    main()
