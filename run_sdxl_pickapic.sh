export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE="madebyollin/sdxl-vae-fp16-fix" # make sure it is not empty
export DATASET_NAME="/nas/xinxuan/model/datasets--liuhuohuo2--pick-a-pic-v2/pickapic" # make sure "pickapic" in dataset_name 
export HF_ENDPOINT=https://hf-mirror.com
LR=1e-9
WORLD_SIZE=1
ACCUMULATION_STEPS=1
BETA=5000
BATCH_SIZE=$((WORLD_SIZE * ACCUMULATION_STEPS))
CACHE_DIR="/nas/zhiyi/huggingface_cache/datasets"
RUN_NAME="SDXL_PickaPic_lr${LR}_bs${BATCH_SIZE}_beta${BETA}"
OUTPUT_DIR="/nas/zhiyi/output/Diffusers_sdxl_dpo/${RUN_NAME}"

accelerate launch --mixed_precision="fp16"  train/train_sd_dpo.py \
  --sdxl \
  --pretrained_vae_model_name_or_path=$VAE \
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
  --checkpointing_steps 10 \
  --validation_steps 10 \
  --beta_dpo 5000 \
  --output_dir=$OUTPUT_DIR \
  --choice_model="pickscore" \
  --tracker_project_name=$RUN_NAME \
  