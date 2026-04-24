from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel, SD3Transformer2DModel, StableDiffusion3Pipeline, QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKL
import torch
import os
from rewards import PickScoreScorer, HPSv2, AestheticScorer, ClipScorer, ImageRewardScorer
from rewards.hpsv3_api import HPSv3Scorer
from argparse import ArgumentParser
import json
from tqdm import tqdm

# Diffusion-DPO: https://huggingface.co/mhdang/dpo-sdxl-text2image-v1
# Diffusion-KTO: https://huggingface.co/jacklishufan/diffusion-kto
# MAPO: https://huggingface.co/mapo-t2i
model_id_mapping= {
    'sd1.5': {
        'ori': 'sd-legacy/stable-diffusion-v1-5',
        'sft': 'path/to/your/own/reproduction',
        'diffusion_dpo': 'mhdang/dpo-sd1.5-text2image-v1',
        'diffusion_kto': 'jacklishufan/diffusion-kto',
        'linear_dpo': 'whynot0128/Linear-DPO',
        'dspo': 'path/to/your/own/reproduction'
    },

    'sdxl': {
        'ori': 'stabilityai/stable-diffusion-xl-base-1.0',
        'diffusion_dpo': 'mhdang/dpo-sdxl-text2image-v1',
        'mapo': 'mapo-t2i/mapo-beta',
        'sft': "path/to/your/own/reproduction",
        'linear_dpo': 'whynot0128/Linear-DPO-SDXL',
    },
    'sd3': {
        'ori': 'stabilityai/stable-diffusion-3-medium-diffusers',
        'sft': 'path/to/your/own/reproduction',
        'diffusion_dpo': 'path/to/your/own/reproduction',
        'linear_dpo': 'whynot0128/Linear-DPO-SD3',
    },
    'qwen': {
        'ori': '/mnt/ramdisk/hf_qwen_image',
        }
}



scorer_mapping = {
    'pickscore': PickScoreScorer,
    'hpsv2': HPSv2,
    'hpsv3': HPSv3Scorer,
    'aesthetic_score': AestheticScorer,
    'clip_score' : ClipScorer,
    'image_reward' : ImageRewardScorer,
}

prompt_path_mapping = {
    'pickapic': 'data/pickapicv2_test500.txt',
    'hpsv2': 'data/hpsv2_test400.txt',
    'partiprompts': 'data/partiprompts_1632.txt',
}

def dummy_checker(images, *args, **kwargs):
    return images, [False] * len(images)

def load_pipeline(model_name, method, dtype=torch.float16, original_model_id=None, variant=None):
    model_name = model_name.lower()
    method = method.lower()
    assert model_name in model_id_mapping.keys() 
    assert method in model_id_mapping[model_name].keys()

    model_id = model_id_mapping[model_name][method]
    if model_name == 'sd1.5':

        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
        original_model_id = "runwayml/stable-diffusion-v1-5" if original_model_id is None else original_model_id
        pipe = StableDiffusionPipeline.from_pretrained(original_model_id, unet=unet, torch_dtype=dtype)

    elif model_name == 'sdxl':
        if method=='mapo':
            unet = UNet2DConditionModel.from_pretrained(model_id, torch_dtype=dtype)
        else:
            unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
        vae = None
        if dtype == torch.float16:
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
            
        original_model_id = "stabilityai/stable-diffusion-xl-base-1.0" if original_model_id is None else original_model_id
        pipe = StableDiffusionXLPipeline.from_pretrained(original_model_id, vae=vae, unet=unet, torch_dtype=dtype, variant=variant)

    elif model_name == 'sd3':
        transformer = SD3Transformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)
        original_model_id = "stabilityai/stable-diffusion-3-medium-diffusers" if original_model_id is None else original_model_id
        pipe = StableDiffusion3Pipeline.from_pretrained(original_model_id, transformer=transformer, torch_dtype=dtype)
    elif model_name == 'qwen':
        transformer = QwenImageTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)
        original_model_id = "Qwen/Qwen-Image" if original_model_id is None else original_model_id
        pipe = QwenImagePipeline.from_pretrained(original_model_id, transformer=transformer, torch_dtype=dtype)

    else: # todo list [qwen-image flux]
        raise NotImplementedError
    pipe.safety_checker = dummy_checker
    pipe.set_progress_bar_config(disable=True)
    return pipe

def load_prompts(prompt_file):
    assert prompt_file.lower() in prompt_path_mapping.keys()
    prompt_file = prompt_path_mapping[prompt_file.lower()]
    with open(prompt_file, 'r') as f:
        prompts = f.readlines()
    return prompts


def load_scorer(scorer_name, device, dtype=torch.float32):
    scorer_name = scorer_name.lower()
    return scorer_mapping[scorer_name](device, dtype)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", choices= ["sd1.5", "sdxl", "sd3", 'qwen'], default="sd1.5")
    parser.add_argument("--method", choices = ["diffusion_dpo", "diffusion_kto", 'mapo','ori', 'linear_dpo', 'dspo', 'sft'], default="diffusion_dpo")
    parser.add_argument("--scorers", type=str, nargs="+", default=["pickscore", "hpsv2", "aesthetic_score", "clip_score", "image_reward", "hpsv3"])
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["pickapic", "hpsv2", "partiprompts"])
    parser.add_argument("--output_dir", type=str, default="../validation_output")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--dtype", choices=['fp16', 'fp32','bf16'], default="fp16")
    parser.add_argument("--original_model_id", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def main(args):
    # makedir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    # prepare pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    pipe = load_pipeline(args.model_name, args.method, dtype, args.original_model_id, args.variant)
    pipe = pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    if args.width is None or args.height is None:
        args.width = 512 if args.model_name == "sd1.5" else 1024
        args.height = 512 if args.model_name == "sd1.5" else 1024
    # prepare scorers
    scorers = [load_scorer(scorer_name, device) for scorer_name in  args.scorers]

    results = {}
    avg_scores = {}
    for benchmark in args.benchmarks:
        print(f"Starting validation for {benchmark}")
        # makedirs
        if not os.path.exists(os.path.join(args.output_dir, "images", benchmark)):
            os.makedirs(os.path.join(args.output_dir, "images", benchmark), exist_ok=True)

        prompts = load_prompts(benchmark)
        rewards = {scorer_name : {} for scorer_name in args.scorers}
        total_scores = {scorer_name : [] for scorer_name in args.scorers}

        for idx in tqdm(range(len(prompts))):
            prompt = prompts[idx].strip()
            images = pipe(
                        prompt, 
                        width=args.width,
                        height=args.height,
                        num_inference_steps=args.num_inference_steps, 
                        num_images_per_prompt=args.num_images_per_prompt, 
                        generator=generator
                        ).images
            [image.save(os.path.join(args.output_dir,"images", benchmark, f"image_{idx}_{i}.jpg")) for i, image in enumerate(images)]
            image_paths = [os.path.join(args.output_dir,"images", benchmark, f"image_{idx}_{i}.jpg") for i in range(len(images))]
            for scorer_name, scorer in zip(args.scorers, scorers):
                if isinstance(scorer, HPSv3Scorer):
                    scores = [scorer(prompt, image_path) for image_path in image_paths]
                else:
                    scores = [scorer(prompt, image)[0] for image in images]
                total_scores[scorer_name].extend(scores)
                rewards[scorer_name][prompt] = scores

        results[benchmark] = rewards
        avg_scores[benchmark] = {scorer_name : torch.tensor(total_scores[scorer_name]).mean().item() for scorer_name in args.scorers}
        print(f"The avg scores of {benchmark} are: ", avg_scores[benchmark])
        # gather results of every benchmark
        with open(os.path.join(args.output_dir, f"{benchmark}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results[benchmark], f, indent=4, ensure_ascii=False)
        with open(os.path.join(args.output_dir, f"{benchmark}_avg_scores.json"), 'w', encoding='utf-8') as f:
            json.dump(avg_scores[benchmark], f, indent=4, ensure_ascii=False)

    # gather all results of all benchmarks
    with open(os.path.join(args.output_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "avg_scores.json"), 'w', encoding='utf-8') as f:
        json.dump(avg_scores, f, indent=4, ensure_ascii=False)
    
            


       
if __name__ == "__main__":
    args = get_args()
    main(args)
    
    
