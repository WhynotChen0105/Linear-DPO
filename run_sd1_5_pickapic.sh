export MODEL_NAME="sd-legacy/stable-diffusion-v1-5"
export DATASET_NAME="./datasets--liuhuohuo2--pick-a-pic-v2/pickapic" # make sure "pickapic" in dataset_name 
export HF_ENDPOINT=https://hf-mirror.com

LR=1e-8
WORLD_SIZE=8
ACCUMULATION_STEPS=256
BETA=2500
BATCH_SIZE=$((WORLD_SIZE * ACCUMULATION_STEPS))
CACHE_DIR="./huggingface_cache/datasets"
RUN_NAME="SD_PickaPic_lr${LR}_bs${BATCH_SIZE}_beta${BETA}"
OUTPUT_DIR="./outputs/Diffusers_sd_dpo/${RUN_NAME}"

accelerate launch --mixed_precision="fp16"  train/train_sd_dpo.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=$ACCUMULATION_STEPS \
  --max_train_steps=100000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=$LR \
  --scale_lr \
  --cache_dir=$CACHE_DIR \
  --checkpointing_steps 300 \
  --validation_steps 300 \
  --beta_dpo=$BETA \
  --output_dir=$OUTPUT_DIR \
  --choice_model="pickscore" \
  --tracker_project_name=$RUN_NAME \
