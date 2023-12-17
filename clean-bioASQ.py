import pandas as pd
import json
import re
import os
import numpy as np


# a csv dataset file with: id; context; question; answers_text; answer_start; ideal_answer; exact_answer; and type
def clean(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    questions = data['questions']

    df = pd.DataFrame(questions)

    data = df[['id', 'body', 'ideal_answer', 'exact_answer', 'type']]
    data = data.rename(columns={'body': 'question'})
    # data.columns = [['id', 'context', 'question', 'answer_text', 'answer_start']]

    def process(x, sec):
        r = []
        for i in x:
            r.append(i[sec])
        return r

    data.loc[:, 'answer_text'] = df.loc[:, 'snippets'].map(
        lambda x: process(x, 'text'))
    data.loc[:, 'answer_start'] = df.loc[:, 'snippets'].map(
        lambda x: process(x, 'offsetInBeginSection'))
    data.loc[:, 'context'] = df.loc[:, 'snippets'].map(
        lambda x: process(x, 'document'))
    data.loc[:, 'exact_answer'] = data.loc[:, 'exact_answer'].map(
        lambda x: (1 if x == 'yes' else 0) if x in ['yes', 'no'] else x)

    data = data.reindex(columns=['id', 'context', 'question', 'answer_text',
                        'answer_start', 'ideal_answer', 'exact_answer', 'type'])

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
