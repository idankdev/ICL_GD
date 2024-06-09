import numpy as np
import torch
from custom_datasets import TextDataLoader
from models import PermutationIntervention
import tqdm
import re

def get_dem_begin_indices(model, icl_prompt):
    '''
    Returns list of indices in token list in which each demonstration of the icl prompt starts (including test prompt)
    '''
    ret = []
    start_idxs = [m.start() for m in re.finditer('\nNews', icl_prompt)]
    for idx in start_idxs:
        tokens = model.tokenizer.encode(icl_prompt[:idx])
        ret.append(len(tokens))
    return ret


def get_preserving_permutation(model, icl_prompt, rng=None):
    '''
    Returns permutation indexes than can be applied to the encoding of icl_prompt in order to shuffle the demonstrations
    '''
    ret = []
    if rng is None:
        rng = np.random.default_rng()
    dem_begin_indices = get_dem_begin_indices(model, icl_prompt)
    dem_order = np.arange(0, len(dem_begin_indices) - 1)  # Not including test prompt
    rng.shuffle(dem_order)
    ret += list(np.arange(0, dem_begin_indices[0]))
    for old_pos in dem_order:
        ret += list(
            np.arange(dem_begin_indices[old_pos], dem_begin_indices[old_pos + 1])
        )
    orig_tokens = model.tokenizer.encode(icl_prompt)
    ret += list(np.arange(dem_begin_indices[-1], len(orig_tokens)))
    assert len(ret) == len(orig_tokens)
    return torch.tensor(ret)

def get_semi_preserving_permutation(model, icl_prompt, seed=None):
    '''
    Returns permutation that jumbles up everything after the initial instruction and before the final test prompt.
    '''
    ret = []
    rng = np.random.default_rng(seed=seed)
    dem_begin_indices = get_dem_begin_indices(model, icl_prompt)
    all_middle_indices = np.arange(start=dem_begin_indices[0], stop=dem_begin_indices[-1])
    rng.shuffle(all_middle_indices)
    ret += list(np.arange(0, dem_begin_indices[0]))
    ret += list(all_middle_indices)
    orig_tokens = model.tokenizer.encode(icl_prompt)
    ret += list(np.arange(dem_begin_indices[-1], len(orig_tokens)))
    assert len(ret) == len(orig_tokens)
    return torch.tensor(ret)

def get_non_preserving_permutation(model, icl_prompt, seed=None):
    '''
    Returns permutation that jumbles up literally everything
    '''
    rng = np.random.default_rng(seed=seed)
    orig_tokens = model.tokenizer.encode(icl_prompt)
    indices = np.arange(0, len(orig_tokens))
    rng.shuffle(indices)
    return torch.tensor(indices)



def model_accuracy(model, ds, test_size, batch_size=1, seed=None, permute=False, permute_layer=None, permute_type='preserving'):
    '''
    Support only batch_size = 1 for now
    '''
    correct = 0
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(ds), size=(test_size,), replace=False)
    sub_ds = ds[indices]
    dl = TextDataLoader(sub_ds, batch_size=batch_size, shuffle=False)
    for i, batch in tqdm(enumerate(dl)):
        prompt, correct_answer = batch
        interventions = []
        if permute:
            assert permute_layer is not None
            if isinstance(permute_layer, int):
                permute_layer = [permute_layer]
            for layer in permute_layer:
                if permute_type == 'preserving':
                    perm = get_random_dem_permutation(model, prompt, seed=seed)
                elif permute_type == 'semi-preserving':
                    perm = get_semi_preserving_permutation(model, prompt, seed=seed)
                elif permute_type == 'non-preserving':
                    perm = get_non_preserving_permutation(model, prompt, seed=seed)
                else:
                    raise NotImplementedError
                interventions.append(PermutationIntervention(perm=perm, before_layer_num=layer))
        logits = model.forward(prompt, interventions=interventions)
        answer = model.tokenizer.decode(torch.argmax(logits[0, -1]))
        correct += (answer == correct_answer)

    return correct / test_size