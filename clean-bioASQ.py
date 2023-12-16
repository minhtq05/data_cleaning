import pandas as pd
import json
import re
import os
import numpy as np


# I will need a csv dataset file with: id; context; question; answers_text; answer_start.
def clean(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    questions = data['questions']

    df = pd.DataFrame(questions)

    data = df[['id', 'snippets', 'body', 'snippets', 'snippets']]
    data.columns = [
        ['id', 'context', 'question', 'answer_text', 'answer_start']]

    def process(x, sec):
        r = []
        for i in x:
            r.append(i[sec])
        return r

    n_rows = len(data)
    data.loc[:, 'answer_text'] = data.loc[:, 'answer_text'].map(
        lambda x: process(x, 'text'))
    data.loc[:, 'answer_start'] = data.loc[:, 'answer_start'].map(
        lambda x: process(x, 'offsetInBeginSection'))
    data.loc[:, 'context'] = data.loc[:, 'context'].map(
        lambda x: process(x, 'document'))

    data.to_csv(file_path[:-4] + 'csv', index=False)


def read_data(path):
    list_dirs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                print(os.path.join(root, file))
                list_dirs.append(os.path.join(root, file))
    if len(list_dirs) == 0:
        print('Oops, nothing found')
    return list_dirs


list_dir = read_data('./data/bioASQ11/')
for file in list_dir:
    clean(file)
