#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE="madebyollin/sdxl-vae-fp16-fix" # make sure it is not empty
export DATASET_NAME="yuvalkirstain/pickapic_v2" # make sure "pickapic" in dataset_name 
export HF_ENDPOINT=https://hf-mirror.com
export WORLD_SIZE=16
export ACCUMULATION_STEPS=8
CACHE_DIR="./huggingface_cache/datasets"

LR_LIST=(1e-5)
BETA_LIST=(500)
ETA_list=(0)
DECAYS=(0.99 0.995 0.95)
for LR in "${LR_LIST[@]}"; do
  for BETA in "${BETA_LIST[@]}"; do
    for ETA in "${ETA_list[@]}"; do
      BATCH_SIZE=$((WORLD_SIZE * ACCUMULATION_STEPS))
      RUN_NAME="linear_SDXL_PickaPic_lr${LR}_bs${BATCH_SIZE}_beta${BETA}_eta${ETA}"
      OUTPUT_DIR="./outputs/Diffusers_sdxl_dpo/${RUN_NAME}"

      accelerate launch --mixed_precision="fp16" train/train_sd_dpo.py \
        --sdxl \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --pretrained_vae_model_name_or_path=$VAE \
        --dataset_name=$DATASET_NAME \
        --train_batch_size=1 \
        --mixed_precision="fp16" \
        --dataloader_num_workers=16 \
        --gradient_accumulation_steps=$ACCUMULATION_STEPS \
        --max_train_steps=2000 \
        --lr_scheduler="constant_with_warmup" --lr_warmup_steps=0 \
        --learning_rate=$LR \
        --cache_dir=$CACHE_DIR \
        --checkpointing_steps 2000 \
        --validation_steps 60 \
        --beta_dpo=$BETA \
        --output_dir=$OUTPUT_DIR \
        --choice_model="pickscore" \
        --tracker_project_name=$RUN_NAME \
        --linear_dpo \
        --eta_dpo=${ETA} \

      echo "✅ Finished: LR=$LR  BETA=$BETA"
    done
  done
done
