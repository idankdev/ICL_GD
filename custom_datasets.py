import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np

class AGNewsICL(Dataset):
    AG_NEWS_LABEL_TO_WORD = {
        0: 'World',
        1: 'Sports',
        2: 'Business',
        3: 'Technology',
    }

    AG_NEWS_INSTRUCTION = 'Classify the news articles into the categories of World, Sports, Business, and Technology.\n'

    def __init__(self, csv_path):
        self.ds = pd.read_csv(csv_path)
        self.name = csv_path
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds.loc[idx, 'ICL Prompt'], self.ds.loc[idx, 'Answer']
    
    @staticmethod
    def create_dataset(n_demonstrations, n_prompts, save_path, seed=None):
        orig_ds = load_dataset('ag_news')['train']
        rng = np.random.default_rng(seed=seed)
        test_indices = rng.choice(len(orig_ds), size=(n_prompts,), replace=False)
        icl_ds = []
        for i in range(n_prompts):
            icl_prompt = AGNewsICL.AG_NEWS_INSTRUCTION
            zsl_prompt = icl_prompt

            test_index = int(test_indices[i])
            test_data_point = orig_ds[test_index]
            indices_wo_test = list(range(len(orig_ds)))
            indices_wo_test.remove(test_index)
            test_text, test_label = test_data_point['text'], AGNewsICL.AG_NEWS_LABEL_TO_WORD[test_data_point['label']]

            dem_indices = rng.choice(indices_wo_test, size=(n_demonstrations, ), replace=False)
            for j in range(n_demonstrations):
                data_point = orig_ds[int(dem_indices[j])]
                text, label = data_point['text'], AGNewsICL.AG_NEWS_LABEL_TO_WORD[data_point['label']]
                icl_prompt += f"News: {text}\nCategory: {label}\n"
            zsl_prompt += f"News: {test_text}\nCategory:"
            icl_prompt += f"News: {test_text}\nCategory:"
            icl_ds.append({'ICL Prompt': icl_prompt, 'ZSL Prompt': zsl_prompt,'Answer': test_label})
        icl_ds = pd.DataFrame(icl_ds)
        icl_ds.to_csv(save_path, index=False)
        return AGNewsICL(save_path)

class AGNewsZSL(Dataset):
    def __init__(self, csv_path):
        self.ds = pd.read_csv(csv_path)
        self.name = csv_path

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds.loc[idx, 'ZSL Prompt'], self.ds.loc[idx, 'Answer']


class TextDataLoader():
    '''
    Custom simple dataloader that returns batches as lists of tuples (prompt, answer)
    '''
    
    def __init__(self, ds, batch_size=1, shuffle=True, seed=None):
        self.ds = ds
        self.rng = np.random.default_rng(seed=seed)
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        self.load_order = np.arange(len(self.ds))
        if self.shuffle:
            self.rng.shuffle(self.load_order)
        self.i = 0
        return self
    
    def __next__(self):
        if self.i < len(self.load_order):
            ret = [self.ds[self.load_order[j]] for j in range(self.i, min(self.i + self.batch_size, len(self.load_order)))]
            self.i += self.batch_size
            return ret
        else:
            raise StopIteration
        


