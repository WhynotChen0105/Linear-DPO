export HF_ENDPOINT=https://hf-mirror.com
MODEL_NAME="/mnt/fuse/.cache/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"
DATASET_NAME="/nas/zhiyi/data/HPDv3" # make sure "pickapic" in dataset_name 
LR=5e-6
WORLD_SIZE=8
ACCUMULATION_STEPS=16
BETA=1000
BATCH_SIZE=$((WORLD_SIZE * ACCUMULATION_STEPS))
CACHE_DIR="/nas/zhiyi/huggingface_cache/datasets"
RUN_NAME="85k_SD3_HPDv3_lr${LR}_bs${BATCH_SIZE}_beta${BETA}"
OUTPUT_DIR="/nas/zhiyi/output/Diffusers_sd3_dpo_new/${RUN_NAME}"

accelerate launch train/train_sd3_dpo.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=$ACCUMULATION_STEPS \
  --max_train_steps=1000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=0 \
  --learning_rate=$LR \
  --cache_dir=$CACHE_DIR \
  --checkpointing_steps 100 \
  --validation_steps 50 \
  --beta_dpo=$BETA \
  --output_dir=$OUTPUT_DIR \
  --choice_model="pickscore" \
  --tracker_project_name=$RUN_NAME \
  --max_train_samples 50000 \