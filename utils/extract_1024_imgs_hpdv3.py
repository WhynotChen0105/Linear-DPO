import os
import json
from multiprocessing import Pool, cpu_count

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ========= 配置部分 =========
DATA_ROOT = "/nas/zhiyi/data/HPDv3"
JSON_PATH = os.path.join(DATA_ROOT, "train.json")
OUTPUT_JSON_PATH = os.path.join(DATA_ROOT, "train_1024.json")
HF_CACHE_DIR = "/nas/zhiyi/huggingface_cache/datasets"

candidate_models = ["flux", "kolors", "sd3", "hunyuan", "real_images"]

# 可根据机器情况调整
NUM_WORKERS = min(100, cpu_count())  # 你也可以直接写 8, 16 等

TARGET_SIZE = (1024, 1024)  # (width, height)


# ========= 1. 加载数据集 & 过滤模型 =========
def load_and_filter_dataset():
    data_files = {"train": JSON_PATH}
    dataset = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=HF_CACHE_DIR,
    )

    orig_len = dataset["train"].num_rows

    candidate_idx = [
        i
        for i, (m0, m1) in enumerate(
            zip(dataset["train"]["model1"], dataset["train"]["model2"])
        )
        if (m0 in candidate_models) and (m1 in candidate_models)
    ]

    dataset["train"] = dataset["train"].select(candidate_idx)
    new_len = dataset["train"].num_rows
    print(f"Eliminated {orig_len - new_len}/{orig_len} non-candidate gens for HPDv3")

    return dataset["train"]


# ========= 2. 多进程：单个样本的处理函数 =========
def check_pair_1024(item):
    """
    多进程 worker 调用的函数。
    输入：一个样本（dict），包含 'path1','path2'
    输出：
      - 如果两张图都存在、尺寸相同且都是 1024x1024，则返回 item 本身；
      - 否则返回 None。
    """
    path1 = os.path.join(DATA_ROOT, item["path1"])
    path2 = os.path.join(DATA_ROOT, item["path2"])

    try:
        with Image.open(path1) as im1:
            size1 = im1.size  # (width, height)
        with Image.open(path2) as im2:
            size2 = im2.size

        # 条件：尺寸相同，且都为 1024x1024
        if size1 == size2 == TARGET_SIZE:
            return item
        else:
            return None
    except Exception as e:
        # 如果图片打不开或有问题，就跳过
        # print(f"Error processing {path1} or {path2}: {e}")
        return None


# ========= 3. 并行筛选 =========
def filter_pairs_1024(train_dataset, num_workers=NUM_WORKERS):
    """
    对 train_dataset 中的所有 pair，使用多进程筛选：
      - 两张图尺寸相同；
      - 且两张图尺寸都是 1024x1024。
    返回：满足条件的样本列表（Python list of dict）。
    """
    total_pairs = train_dataset.num_rows
    print(f"Total pairs after model filtering: {total_pairs}")

    # 为减少 Dataset 单条索引的开销，先转换成 list[dict]
    items = [train_dataset[i] for i in range(total_pairs)]

    print(f"Using {num_workers} worker processes for multiprocessing...")

    valid_items = []

    with Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap_unordered(check_pair_1024, items), total=total_pairs):
            if res is not None:
                valid_items.append(res)

    print(f"Kept {len(valid_items)}/{total_pairs} pairs with both images == {TARGET_SIZE}")
    return valid_items


# ========= 4. 写出新的 JSON =========
def save_to_json(items, output_path):
    """
    items: list of dict（每个 dict 就是一条样本，字段跟原 train.json 保持一致）
    """
    # 直接写成一个 list，和原 train.json 格式一致（如果原来就是 list of dict）
    # 如果原来是 {"train": [...] }，你也可以改成那种格式。
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"Saved {len(items)} items to {output_path}")


# ========= 5. 主程序入口 =========
def main():
    # 1) 先过滤模型条件
    train_dataset = load_and_filter_dataset()

    # 2) 并行筛选两张图都为 1024x1024 且尺寸相同的样本对
    valid_items = filter_pairs_1024(train_dataset, num_workers=NUM_WORKERS)

    # 3) 写出新的 train_1024.json
    save_to_json(valid_items, OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()
