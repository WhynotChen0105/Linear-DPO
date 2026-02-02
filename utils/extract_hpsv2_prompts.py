from huggingface_hub import hf_hub_download
import json

file_path = hf_hub_download(
    repo_id="ymhao/HPDv2",
    filename="test.json",           
    repo_type="dataset",
    local_dir="data/HPDv2_data"         
)

data = json.load(open(file_path))
prompts = []
for item in data:
    prompts.append(item['prompt'])

with open('data/hpsv2.txt', 'w') as f:
    for prompt in prompts:
        f.write(prompt + '\n')