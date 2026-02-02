from datasets import load_dataset


test_dataset = load_dataset(
                "parquet",
                data_files={
                    # "validation_unique": "/nas/zhiyi/data/PickaPic/datasets--pickapic-anonymous--pickapic_v1/snapshots/8e69b58d729339cb7771133c70ee18b527fd4ae3/data/validation_unique*.parquet",
                    "test_unique": "/nas/xinxuan/model/datasets--liuhuohuo2--pick-a-pic-v2/pickapic/data/test_unique*.parquet",
                })
test_prompts = test_dataset['test_unique']['caption']
test_prompts = [prompt for prompt in test_prompts if prompt is not None]

with open('data/pickapicv2_test500.txt', 'w') as f:
    for prompt in test_prompts:
        f.write(prompt + '\n')