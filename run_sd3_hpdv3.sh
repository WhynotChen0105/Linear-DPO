#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# 固定参数
MODEL_NAME="/mnt/fuse/.cache/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"
DATASET_NAME="/zjk_nas/zhiyi/data/HPDv3" # make sure "pickapic" in dataset_name
WORLD_SIZE=8
ACCUMULATION_STEPS=16
BATCH_SIZE=$((WORLD_SIZE * ACCUMULATION_STEPS))
CACHE_DIR="/zjk_nas/zhiyi/huggingface_cache/datasets"
MAX_TRAIN_STEPS=3200
MIXED_PRECISION="fp16"
DATALOADER_WORKERS=8

LR_LIST=(5e-6)
BETA_LIST=(5000)
ETA_list=(5e-2)
DECAY=(0.99)

for LR in "${LR_LIST[@]}"; do
    for BETA in "${BETA_LIST[@]}"; do
        for ETA in "${ETA_list[@]}"; do
          for DECAY in "${DECAY[@]}"; do
            RUN_NAME="SD3_HPDv3_lr${LR}_bs${BATCH_SIZE}_beta${BETA}"
            OUTPUT_DIR="/zjk_nas/zhiyi/output/Diffusers_sd3_dpo_ema/${RUN_NAME}"
            
            echo "=============================="
            echo "🚀 开始训练: LR=${LR}, BETA=${BETA}, ETA=${ETA}"
            echo "Run name: $RUN_NAME"
            echo "输出目录: $OUTPUT_DIR"
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
              --eta_dpo="$ETA" \
              --output_dir="$OUTPUT_DIR" \
              --choice_model="hpsv3" \
              --tracker_project_name="$RUN_NAME" \

          done
        done
    done
done
