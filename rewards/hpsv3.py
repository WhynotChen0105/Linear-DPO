from src.hpsv3 import HPSv3RewardInferencer
import torch

class HPSv3Scorer():
    def __init__(self, device='cuda', dtype='float32'):
        self.inferencer = HPSv3RewardInferencer(device=device)

    @torch.no_grad()
    def __call__(prompts, images):
        if not isinstance(images, list):
            images = [images]
        if not isinstance(prompts, list):
            prompts = [prompts]
        rewards = self.inferencer.reward(prompts, image_paths=images)
        scores = [reward[0].item() for reward in rewards]
        return scores
