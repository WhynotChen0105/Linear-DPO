import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from rewards.hpsv3_api import HPSv3Scorer
from transformers.utils import ContextManagers
import copy
from typing import Optional, Iterator
import torch.nn as nn


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    step,
    torch_dtype,
    scorer=None,
    ema_model=False,
    negative_prompt_embeds=None, 
    negative_prompt_embeds_mask=None,
    is_final_validation=False,
):  

    # args.num_validation_images = args.num_validation_images if args.num_validation_images else 1
    args.num_validation_images = 1
    
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    def dummy_checker(images, *args, **kwargs):
        return images, [False] * len(images)
    pipeline.safety_checker = dummy_checker

    # distributed sampling
    num_validation_samples = len(pipeline_args)
    num_processes = accelerator.num_processes
    current_process_index = accelerator.process_index
    samples_per_process = num_validation_samples // num_processes
    remainder = num_validation_samples % num_processes
    start_idx = current_process_index * samples_per_process + min(current_process_index, remainder)
    end_idx = (current_process_index + 1) * samples_per_process + min(current_process_index + 1, remainder)
    local_indices = list(range(start_idx, end_idx))

    for idx in local_indices:
        with torch.no_grad():
            with autocast_ctx:
                print(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {pipeline_args[idx]}."
                )
                image = pipeline(
                    prompt = pipeline_args[idx],
                    width=args.resolution,
                    height=args.resolution,
                    generator=generator,
                ).images[0]
                image_name = f"image_{idx}_step{step}_ema.jpg" if ema_model else f"image_{idx}_step{step}.jpg"
                save_path = os.path.join(args.output_dir, "images", image_name)
                image.save(save_path)
    
    accelerator.wait_for_everyone()
    # gather and upload images
    if accelerator.is_main_process:
        reward_score = 0.0
        success_rewards = 0
        # Get Reward Scores
        if args.choice_model:
            for idx in tqdm(range(len(pipeline_args))):
                if isinstance(scorer, HPSv3Scorer):
                    score = scorer(pipeline_args[idx], os.path.join(args.output_dir.replace("/zjk_nas/", "/nas/"), "images", f"image_{idx}_step{step}.jpg"))
                else:
                    image_name = f"image_{idx}_step{step}_ema.jpg" if ema_model else f"image_{idx}_step{step}.jpg"
                    with Image.open(os.path.join(args.output_dir, "images", image_name)) as image:
                        score = scorer(pipeline_args[idx], image)
                        if isinstance(score, list):
                            score = score[0]
                if score is not None:
                    success_rewards += 1
                    reward_score += score
            reward_score /= success_rewards
            print(f"Average Score: {reward_score}, Success Rate: {success_rewards}/{len(pipeline_args)}")
        # logging
        for tracker in accelerator.trackers:
            if ema_model:
                phase_name = "test_ema" if is_final_validation else "validation_ema"
            else:
                phase_name = "test" if is_final_validation else "validation"
            scalar_data_to_log = {
            f"{phase_name}/average_score": reward_score}

            if tracker.name == "tensorboard":
                for tag, value in scalar_data_to_log.items():
                    if isinstance(value, (int, float, torch.Tensor, np.number)):
                        tracker.writer.add_scalar(tag, value, step)
                    else:
                        print(f"Warning: Skipping non-scalar value for TensorBoard: {tag}={value}")
                tracker.writer.flush()

            elif tracker.name == "wandb":
                tracker.log(scalar_data_to_log, step=step)
            else:
                print(f"Warning: Unknown tracker: {tracker.name}")
    
    return None


class ModelEMA(nn.Module):
    """
    EMA wrapper that keeps a separate deep-copied model and updates its
    parameters in-place using an exponential moving average.

    - On init: deep copies `model` -> `self.ema_model`
    - On update(): directly updates `self.ema_model` parameters (and buffers if enabled)
    - Call: `ema(x)` == `ema.ema_model(x)`
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
        use_num_updates: bool = False,
        requires_grad: bool = False,
        update_buffers: bool = True,
    ):
        super().__init__()

        self.ema_model = copy.deepcopy(model)

        if not requires_grad:
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        if device is not None:
            self.ema_model.to(device)

        self.decay = decay
        self.use_num_updates = use_num_updates
        self.update_buffers = update_buffers
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA model from `model`. Call this after optimizer.step().
        """
        self.num_updates += 1

        decay = self.decay
        if self.use_num_updates:
            # Simple warmup-style scheduling; adjust as you like
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        # Update all parameters
        ema_params = list(self.ema_model.parameters())
        model_params = list(model.parameters())

        for ema_p, model_p in zip(ema_params, model_params):
            ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)

        # Optionally update all buffers (e.g., BatchNorm running stats)
        if self.update_buffers:
            ema_buffers = list(self.ema_model.buffers())
            model_buffers = list(model.buffers())

            for ema_b, model_b in zip(ema_buffers, model_buffers):
                ema_b.data.copy_(model_b.data)

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    # Convenience wrappers
    def to(self, *args, **kwargs):
        self.ema_model.to(*args, **kwargs)
        return self

    def cuda(self, device: Optional[int] = None):
        self.ema_model.cuda(device)
        return self

    def cpu(self):
        self.ema_model.cpu()
        return self

    def state_dict(self, *args, **kwargs):
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        self.ema_model.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return self.ema_model.parameters(recurse=recurse)

    def eval(self):
        self.ema_model.eval()
        return self

    def train(self, mode: bool = True):
        self.ema_model.train(mode)
        return self

import torch

def get_specific_lora_weights(self, model: torch.nn.Module, adapter_name: str) -> dict:

    lora_state_dict = {}

    for name, module in model.named_modules():
        
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            
            if adapter_name in module.lora_A:
                
                lora_A_module = module.lora_A[adapter_name]
                lora_B_module = module.lora_B[adapter_name]
                
                key_A = f"{name}.lora_A.{adapter_name}.weight"
                key_B = f"{name}.lora_B.{adapter_name}.weight"

                lora_state_dict[key_A] = lora_A_module.weight
                lora_state_dict[key_B] = lora_B_module.weight
                
    return lora_state_dict


def update_lora_ema(self, model, ema_dict, decay, adapter_name):

    current_lora_weights = self.get_specific_lora_weights(model, adapter_name)
    
    for name, param in current_lora_weights.items():
        if name in ema_dict:

            ema_dict[name].lerp_(param.data, weight=1.0 - decay)
        
            
        else:
            ema_dict[name] = param.data.clone()
