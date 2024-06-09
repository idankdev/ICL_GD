import torch
from custom_datasets import AGNewsICL
from transformers import AutoModelForCausalLM
from models import Llama7BHelper
from tqdm import tqdm
from collections import defaultdict


LLAMA_2_7B_PATH = 'meta-llama/Llama-2-7b-hf'

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AGNewsICL('ICL-AGNews-Size1000-8Dem.csv')

model = Llama7BHelper(AutoModelForCausalLM.from_pretrained(LLAMA_2_7B_PATH).to(device), device=device)

accuracies = defaultdict(float)
prob_avg = defaultdict(float)

for idx in tqdm(range(len(dataset))):
    prompt, correct_answer = dataset[idx]
    correct_answer_token = model.tokenizer.encode(text=correct_answer, add_special_tokens=False)[0]
    logits = model.forward(prompt).cpu()
    probs = model.get_prob_lens().cpu() # (layer, pos, d_vocab)
    for layer in range(32):
        prob_avg[f"layer_{layer}_prob_avg"] += probs[layer, -1, correct_answer_token].item()
        accuracies[f"layer_{layer}_accuracy"] += torch.argmax(probs[layer, -1]).item() == correct_answer_token

for key in accuracies:
    accuracies[key] /= len(dataset)

for key in prob_avg:
    prob_avg[key] /= len(dataset)

with open("cut_layer_results.csv", "w") as f:
    f.write("layer,accuracy,prob_avg\n")
    for i in range(32):
        f.write(f"{i},{accuracies[f'layer_{i}_accuracy']},{prob_avg[f'layer_{i}_prob_avg']}\n")
