#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# Fixed parameters
MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
DATASET_NAME="./data/HPDv3" 
WORLD_SIZE=16
ACCUMULATION_STEPS=128
BATCH_SIZE=$((WORLD_SIZE * ACCUMULATION_STEPS))
CACHE_DIR="./huggingface_cache/datasets"
MAX_TRAIN_STEPS=2000
MIXED_PRECISION="fp16"
DATALOADER_WORKERS=8

# Define grid parameter list
LR_LIST=(1e-8)
BETA_LIST=(5000)
ETA_list=(0)
DECAY=(1)
# Iterate over all combinations of learning rate and beta
for LR in "${LR_LIST[@]}"; do
    for BETA in "${BETA_LIST[@]}"; do
        for ETA in "${ETA_list[@]}"; do
          for DECAY in "${DECAY[@]}"; do
            RUN_NAME="DPO_SD3_HPDv3_lr${LR}_bs${BATCH_SIZE}_beta${BETA}_eta${ETA}_ema_${DECAY}"
            OUTPUT_DIR="./outputs/Diffusers_sd3_dpo_ema/${RUN_NAME}"
            
            echo "=============================="
            echo "🚀 Start training: LR=${LR}, BETA=${BETA}, ETA=${ETA}"
            echo "Run name: $RUN_NAME"
            echo "Output directory: $OUTPUT_DIR"
            echo "=============================="

            accelerate launch train/train_sd3_dpo.py \
              --pretrained_model_name_or_path="$MODEL_NAME" \
              --train_data_dir="$DATASET_NAME" \
              --train_batch_size=1 \
              --split="train" \
              --mixed_precision="$MIXED_PRECISION" \
              --dataloader_num_workers="$DATALOADER_WORKERS" \
              --gradient_accumulation_steps="$ACCUMULATION_STEPS" \
              --max_train_steps="$MAX_TRAIN_STEPS" \
              --lr_scheduler="constant_with_warmup" \
              --lr_warmup_steps=100 \
              --resolution=1024 \
              --learning_rate="$LR" \
              --cache_dir="$CACHE_DIR" \
              --checkpointing_steps 100 \
              --validation_steps 100 \
              --beta_dpo="$BETA" \
              --output_dir="$OUTPUT_DIR" \
              --choice_model="pickscore" \
              --tracker_project_name="$RUN_NAME" \

            echo "✅ Finished: LR=${LR}, BETA=${BETA}, ETA=${ETA}, DECAY=${DECAY}"
          done
        done
    done
done
