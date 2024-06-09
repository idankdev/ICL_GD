import torch
from custom_datasets import AGNewsICL
from transformers import AutoModelForCausalLM
from models import Llama7BHelper
from interventions import LearningRateIntervention
from itertools import product
from tqdm import tqdm

LLAMA_2_7B_PATH = 'meta-llama/Llama-2-7b-hf'

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AGNewsICL('ICL-AGNews-Size1000-8Dem.csv')

model = Llama7BHelper(AutoModelForCausalLM.from_pretrained(LLAMA_2_7B_PATH).to(device), device=device)

configurations = [{"start_layer_num": 0, "lr_scale": 1.0}] + [
    {"start_layer_num": start_layer_num, "lr_scale": lr_scale}
    for start_layer_num, lr_scale in product([0, 10, 15, 20, 25, 30], [0.5, 0.75, 1.25, 1.5, 1.75, 2.0])
]

results = []

for config in tqdm(configurations):
    num_correct = 0
    interventions = [LearningRateIntervention(**config)]
    for i in tqdm(range(len(dataset)), leave=False):
        prompt, correct_answer = dataset[i]
        logits = model.forward(prompt, interventions=interventions).cpu()
        answer = model.tokenizer.decode(torch.argmax(logits[0, -1]))
        if answer == correct_answer:
            num_correct += 1
    results.append({"start_layer_num": config["start_layer_num"], "lr_scale": config["lr_scale"], "accuracy": num_correct / len(dataset)})

with open("learning_rate_results.csv", "w") as f:
    f.write("start_layer_num,lr_scale,accuracy\n")
    for result in results:
        f.write(f"{result['start_layer_num']},{result['lr_scale']},{result['accuracy']}\n")
