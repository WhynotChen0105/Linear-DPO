#!/bin/bash

export MODEL_NAME="sd-legacy/stable-diffusion-v1-5"
export DATASET_NAME="./datasets--liuhuohuo2--pick-a-pic-v2/pickapic" # make sure "pickapic" in dataset_name 
export HF_ENDPOINT=https://hf-mirror.com
export WORLD_SIZE=32
export ACCUMULATION_STEPS=4
CACHE_DIR="./huggingface_cache/datasets"

LR_LIST=(5e-6)
BETA_LIST=(100)
ETA_list=(1e-3)
DECAYS=(0.99 0.995 0.95)
for LR in "${LR_LIST[@]}"; do
  for BETA in "${BETA_LIST[@]}"; do
    for ETA in "${ETA_list[@]}"; do
      for DECAY in "${DECAY[@]}"; do
        BATCH_SIZE=$((WORLD_SIZE * ACCUMULATION_STEPS))
        RUN_NAME="linear_85k_SD_PickaPic_lr${LR}_bs${BATCH_SIZE}_beta${BETA}_eta${ETA}_ema${DECAY}"
        OUTPUT_DIR="./outputs/Diffusers_sd_dpo/${RUN_NAME}"

        echo "-------------------------------------------"
        echo "▶ start training: LR=$LR  BETA=$BETA"
        echo "-------------------------------------------"

        accelerate launch --mixed_precision="fp16" train/train_sd_dpo.py \
          --pretrained_model_name_or_path=$MODEL_NAME \
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
          --use_ema_ref \
          --decay_ema=${DECAY} \
          --valid_ema \
          --max_train_samples 85000 \

      done
    done
  done
done
