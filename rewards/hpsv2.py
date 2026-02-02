# from https://github.com/tgxs002/HPSv2/blob/master/hpsv2/img_score.py

hps_version_map = {
    "v2.0": "HPS_v2_compressed.pt",
    "v2.1": "HPS_v2.1_compressed.pt",
}

import torch
from PIL import Image
from src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse
import os
import requests
from typing import Union
import huggingface_hub
warnings.filterwarnings("ignore", category=UserWarning)

class HPSv2:
    def __init__(self, device: str = 'cuda', dtype: torch.dtype = torch.float32, hps_version: str = "v2.0") -> None:
        self.device = device
        self.model, self.preprocess_train, self.preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
        
        checkpoint = torch.load(cp, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = get_tokenizer('ViT-H-14')
        self.model = self.model.to(device)
        self.model.eval()

    def __call__(self, prompt: str, img_path: Union[list, str, Image.Image]) -> list:

        if isinstance(img_path, list):
            result = []
            for one_img_path in img_path:
                # Load your image and prompt
                with torch.no_grad():
                    # Process the image
                    if isinstance(one_img_path, str):
                        image = self.preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=self.device, non_blocking=True)
                    elif isinstance(one_img_path, Image.Image):
                        image = self.preprocess_val(one_img_path).unsqueeze(0).to(device=self.device, non_blocking=True)
                    else:
                        raise TypeError('The type of parameter img_path is illegal.')
                    # Process the prompt
                    text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                    # Calculate the HPS
                    with torch.cuda.amp.autocast():
                        outputs = self.model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T

                        hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                result.append(hps_score[0])    
            return result
        
        elif isinstance(img_path, str):
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                image = self.preprocess_val(Image.open(img_path)).unsqueeze(0).to(device=self.device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                # Calculate the HPS
                with torch.cuda.amp.autocast():
                    outputs = self.model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            return [hps_score[0]]
        elif isinstance(img_path, Image.Image):
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                image = self.preprocess_val(img_path).unsqueeze(0).to(device=self.device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                # Calculate the HPS
                
                outputs = self.model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().tolist()
            return hps_score
        else:
            raise TypeError('The type of parameter img_path is illegal.')
        
