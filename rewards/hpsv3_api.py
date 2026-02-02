import base64
import requests
import json
import os
import torch

class HPSv3Scorer():
    def __init__(self, device="cuda", dtype=torch.float32):
        self.api_url = "http://33.200.227.184:8000/infer_score"

    def __call__(self, prompt, image_path):
        payload = {
        "image_path": image_path,
        "prompt": prompt,
        }
        # 3. 发送 POST 请求
        for _ in range(20):
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                score = response.json()["score"]
                return score
            else:
                print(f"❌ 请求失败，状态码: {response.status_code}")
                print(f"错误详情: {response.json()}")
                continue

        return None

if __name__ == "__main__":
    scorer = HPSv3Scorer(
        device="cuda",
        dtype=torch.float32
    )
    scorer("A race car", "/nas/zhiyi/output/Diffusers_sd3_dpo_new/85k_SD3_HPDv3_lr5e-6_bs128_beta500/images/image_1_step100.jpg")