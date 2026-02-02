MODEL_NAME="sd1.5"
METHOD="diffusion_dpo"
SEED=0

python validation/validate.py \
    --model_name=$MODEL_NAME \
    --method=$METHOD \
    --benchmarks pickapic partiprompts hpsv2 \
    --scorers pickscore hpsv2 aesthetic_score clip_score image_reward \
    --seed=$SEED