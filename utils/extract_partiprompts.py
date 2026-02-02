from datasets import load_dataset

dataset = load_dataset("nateraw/parti-prompts", split='train')

prompts = []
for data in dataset:
    print(data)
    prompts.append(data['Prompt'])

with open('data/partiprompts.txt', 'w') as f:
    for prompt in prompts:
        f.write(prompt + '\n')