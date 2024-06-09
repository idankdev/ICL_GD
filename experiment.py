from custom_datasets import AGNewsICL, AGNewsZSL, TextDataLoader
from utils import get_preserving_permutation
from models import Llama7BHelper
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
from interventions import PermutationIntervention
from torch.multiprocessing.spawn import spawn
import datetime
import os
import time
import torch
import argparse
import pandas as pd
import secrets
import numpy as np

# from torch.multiprocessing import Queue
# AGNewsICL.create_dataset(n_demonstrations=8, n_prompts=10, save_path='ICL-AGNews-Size10-8Dem.csv', seed=493716)
# exit(0)


N_DEVICES = torch.cuda.device_count()
print(f'Devices: {N_DEVICES}')
# assert N_DEVICES == 2
DEVICES = [torch.device(f'cuda:{i}') for i in range(N_DEVICES)]
LLAMA_2_7B_PATH = 'meta-llama/Llama-2-7b-hf'
EXPERIMENT_BASE_PATH = '2024-03-26-Exp1000Prompts'
DATASET = 'ICL-AGNews-Size1000-8Dem.csv'
os.makedirs(EXPERIMENT_BASE_PATH, exist_ok=True)
# SEED = secrets.randbits(32)

def print_wtime(s):
    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]: {s}', flush=True)


def run_permutation_icl(ds, device, interventions, save_path, results_columns=''):
    '''
    Interventions: list of list of interventions (each inner list is tested together on the whole dataset)
    For no interventions, pass [[]]
    '''
    results = []
    print_wtime('Loading model...')
    model = Llama7BHelper(AutoModelForCausalLM.from_pretrained(LLAMA_2_7B_PATH).to(device), device=device)
    for intrv_list in interventions:
        assert intrv_list == [] or all([isinstance(intrv, PermutationIntervention) for intrv in intrv_list])
        total_correct = 0
        intrv_summary = 'No Perm'
        columns = {}
        if results_columns == 'Perm From Layer':
            columns['Perm From Layer'] = min([intrv.before_layer_num for intrv in intrv_list])
            intrv_summary = f'Perm From Layer {columns['Perm From Layer']}'
        elif results_columns == 'Perm In Layer':
            columns['Perm In Layer'] = intrv_list[0].before_layer_num
            intrv_summary = f'Perm In Layer {columns['Perm In Layer']}'
        print_wtime(f'Running {intrv_summary}, num of intrv: {len(intrv_list)}')
        for i in range(len(ds)):
            if i > 0 and i % 100 == 0:
                print_wtime(f'{i} Prompts, Accuracy so far: {total_correct / i}')
            model.reset_all()
            seed = secrets.randbits(32)
            rng = np.random.default_rng(seed)
            prompt, correct_answer = ds[i]
            correct_answer_token = model.tokenizer.encode(text=correct_answer, add_special_tokens=False)[0]
            for intrv in intrv_list:
                intrv.perm = get_preserving_permutation(model, prompt, rng=rng)
            logits = model.forward(prompt, interventions=intrv_list).cpu()
            probs = model.get_prob_lens().cpu()
            answer = model.tokenizer.decode(torch.argmax(logits[0, -1]))
            prompt_columns = {**columns, **{f'Layer {i} confidence': float(probs[i][-1][correct_answer_token]) for i in range(len(model.layers))}}
            correct = answer == correct_answer
            total_correct += correct
            results.append(
                {
                    'DS Name': ds.name,
                    'DS Index': i,
                    'Correct': int(correct),
                    **prompt_columns,
                    'Perms Seed': seed,
                }
            )
        print_wtime(f'Total Accuracy for {intrv_summary}: {total_correct / len(ds)}')
    results = pd.DataFrame(results)
    results.to_csv(save_path)


parser = argparse.ArgumentParser()
parser.add_argument('--experiment-type', choices=['ZSL', 'NoPerm', 'OneLayerPerm', 'FromLayerPerm'])
args = parser.parse_args()


if args.experiment_type == 'ZSL':
    ds = AGNewsZSL(DATASET)
else:
    ds = AGNewsICL(DATASET)
one_layer_permutations = [
    [PermutationIntervention(before_layer_num=i)] for i in range(Llama7BHelper.N_LAYERS)
]
from_layer_permutations = [
    [PermutationIntervention(before_layer_num=j) for j in range(i, Llama7BHelper.N_LAYERS)] for i in range(Llama7BHelper.N_LAYERS)
]

if args.experiment_type == 'ZSL':
    run_permutation_icl(ds, DEVICES[0], [[]], os.path.join(EXPERIMENT_BASE_PATH, 'ZSL.csv'), '')
if args.experiment_type == 'NoPerm':
    run_permutation_icl(ds, DEVICES[0], [[]], os.path.join(EXPERIMENT_BASE_PATH, 'NoPerm.csv'), '')
elif args.experiment_type == 'OneLayerPerm':
    run_permutation_icl(ds, DEVICES[0], one_layer_permutations, os.path.join(EXPERIMENT_BASE_PATH, 'OneLayerPerm.csv'), 'Perm In Layer')
elif args.experiment_type == 'FromLayerPerm':
    run_permutation_icl(ds, DEVICES[0], from_layer_permutations, os.path.join(EXPERIMENT_BASE_PATH, 'FromLayerPerm.csv'), 'Perm From Layer')

# contexts = []
# contexts.append(
#     spawn(, join=False))
# contexts.append(
#     spawn(run_permutation_icl, args=(ds, DEVICES[1], one_layer_permutations, os.path.join(EXPERIMENT_BASE_PATH, 'OneLayerPerm.csv'), 'Perm In Layer', SEED), join=False)
# )
# while not contexts[0].join(timeout=1):  # Wait for first device to finish
#     time.sleep(10)
# contexts.append(
#     spawn(run_permutation_icl, args=(ds, DEVICES[0], from_layer_permutations, os.path.join(EXPERIMENT_BASE_PATH, 'FromLayerPerm.csv'), 'Perm From Layer', SEED), join=False)
# )
# while True:
#     all_finished = all([ctx.join(timeout=1) for ctx in contexts])
#     if all_finished:
#         print_wtime('Finished Experiment!')
#         break
#     time.sleep(10)
