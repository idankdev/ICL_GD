import torch
from custom_datasets import AGNewsICL
from transformers import AutoModelForCausalLM
from models import Llama7BHelper
from tqdm import tqdm
from collections import defaultdict
from interventions import SkipBlockIntervention

LLAMA_2_7B_PATH = 'meta-llama/Llama-2-7b-hf'
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = AGNewsICL('ICL-AGNews-Size1000-8Dem.csv')
model = Llama7BHelper(AutoModelForCausalLM.from_pretrained(LLAMA_2_7B_PATH).to(device), device=device)
SEED = 42
torch.manual_seed(SEED)

results = defaultdict(float) # key: k, value: avg accuracy
for k in tqdm(range(1, 6)):
    interventions = []
    layers_to_remove = torch.randint(15, 32, (k,))
    for layer in layers_to_remove:
        interventions.append(SkipBlockIntervention(layer))
    num_correct = 0
    for idx in tqdm(range(len(dataset))):
        prompt, correct_answer = dataset[idx]
        logits = model.forward(prompt, interventions).cpu()
        answer = model.tokenizer.decode(torch.argmax(logits[0, -1]))
        num_correct += answer == correct_answer
    results[k] = num_correct / len(dataset)

with open("skip_blocks_results.csv", "w") as f:
    f.write("k,avg_accuracy\n")
    for k in results:
        f.write(f"{k},{results[k]}\n")
