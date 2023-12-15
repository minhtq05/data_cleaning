import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class MedQuestionDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        context = self.data.iloc[i]['context']
        question = self.data.iloc[i]['question']
        answer_options = self.data.iloc[i]['answer_option']
        if ('answer' in self.data.columns):
            answer = self.data.iloc[i]['answer']
        else:
            answer = None

        sample = {
            'context': context,
            'question': question,
            'answer_option': answer_options,
            'answer': answer,
        }

        return sample


dataset = MedQuestionDataset('data/ori_pqal.csv', 'data/')


BATCH_SIZE = 100
SHUFFLE = True

dataloader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
)
