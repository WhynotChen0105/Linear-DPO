# Linear-DPO

The official repository for Linear-DPO:LinearDirectPreferenceOptimizationforDiffusionand Flow-MatchingGenerativeModels.

## 🛠️ Setup Environment

First, create a new conda environment with Python 3.11 and activate it:
```bash
conda create -n linear-dpo python=3.11 -y
conda activate linear-dpo
```

Next, install PyTorch (2.2.1) with CUDA 11.8 support:
```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

Finally, install the remaining required dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Prepare Data

### Pick-a-Pic v2 (for SD 1.5 and SDXL)
By default, the training scripts use the Hugging Face dataset `yuvalkirstain/pickapic_v2`. It will be downloaded automatically when you start training.

### HPDv3 (for SD 3)
For SD3 training, we use the HPDv3 dataset. 
You can download the dataset directly to `./data/HPDv3` using the `huggingface-cli`:
```bash
huggingface-cli download MizzenAI/HPDv3 --local-dir ./data/HPDv3 --repo-type dataset
```

### Evaluation Prompts
To extract evaluation prompts for validation, you can use the scripts provided in the `utils/` directory:
```bash
python utils/extract_pickapicv2_prompts.py
python utils/extract_hpsv2_prompts.py
python utils/extract_partiprompts.py
```
*Note: Make sure to check and adjust the paths inside these scripts based on your local dataset locations if you have downloaded them manually.*

## 🏆 Prepare Reward Models

Linear-DPO utilizes several reward models for validation, including:
- **PickScore**
- **HPSv2 / HPSv3**
- **AestheticScore**
- **CLIPScore**
- **ImageReward**

The weights for these reward models will be **automatically downloaded** from Hugging Face to your local cache (e.g., `~/.cache/`) upon their first initialization. You do not need to download them manually. If you wish to use local weights, you can adjust the respective files in the `rewards/` directory.

## 🚀 Train

Before starting the training, you need to configure your environment using `accelerate`. We recommend training with **8 GPUs** to match our default settings.

Run the following command and follow the prompts to generate your configuration file:
```bash
accelerate config
```

We provide shell scripts to easily launch distributed training with `accelerate`.

### Stable Diffusion v1.5
To train SD 1.5 using Linear-DPO on Pick-a-Pic:
```bash
bash run_sd1_5_pickapic_linear.sh
```
*(For standard DPO training, use `run_sd1_5_pickapic.sh`)*

### Stable Diffusion XL (SDXL)
To train SDXL using Linear-DPO on Pick-a-Pic:
```bash
bash run_sdxl_pickapic_linear.sh
```
*(For standard DPO training, use `run_sdxl_pickapic.sh`)*

### Stable Diffusion 3 (SD3)
To train SD3 using Linear-DPO on HPDv3:
```bash
bash run_sd3_hpdv3_linear.sh
```
*(We also provide `run_sd3_hpdv3.sh` for standard DPO and `run_sd3_pickapic.sh` for Pick-a-Pic)*

## 📈 Validation

To evaluate your trained models, you can use the validation script. The script runs inference on benchmarks like `pickapic`, `partiprompts`, and `hpsv2` and scores the generated images using multiple reward models.

To run the validation via the provided shell script:
```bash
bash test.sh
```

Inside `test.sh`, it executes `validation/validate.py`. You can modify `test.sh` or the `model_id_mapping` inside `validation/validate.py` to point to your newly trained model checkpoints and specify which benchmarks/scorers to use.

### Test with Pre-trained Linear-DPO Weights
We provide official pre-trained Linear-DPO weights for SD1.5, SDXL, and SD3. You can test them using the validation script directly:

```bash
python validation/validate.py \
    --model_name sd1.5 \
    --method linear_dpo \
    --benchmarks pickapic partiprompts hpsv2 \
    --scorers pickscore hpsv2 aesthetic_score clip_score image_reward
```
*(For SDXL or SD3, simply change `--model_name sdxl` or `--model_name sd3` respectively.)*
