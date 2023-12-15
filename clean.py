import json
import os
import pandas as pd


def read_data(path):
    list_dirs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jsonl'):
                print(os.path.join(root, file))
                list_dirs.append(os.path.join(root, file))
    return list_dirs


# PubMedQA
def read(p): return pd.read_json(f'./dataset/PubMedQA/ori_pqa{p}.json')


pubmed_l, pubmed_u, pubmed_a = read('l'), read('u'), read('a')


def df_loc(df, ans): return df.loc[[
    'CONTEXTS', 'QUESTION', 'final_decision' if ans == True else 'LONG_ANSWER']]


pubmed_l, pubmed_u, pubmed_a = df_loc(pubmed_l, True), df_loc(
    pubmed_u, False), df_loc(pubmed_a, True)
# pubmed_l = pubmed_l.loc[['CONTEXTS', 'QUESTION', 'final_decision']]
# pubmed_u = pubmed_u.loc[['CONTEXTS', 'QUESTION', 'LONG_ANSWER']]
# pubmed_a = pubmed_a.loc[['CONTEXTS', 'QUESTION', 'final_decision']]

pubmed_l, pubmed_u, pubmed_a = pubmed_l.T, pubmed_u.T, pubmed_a.T


def ans_opt(df): return [['yes', 'no', 'maybe']] * df.shape[0]


pubmed_l['answer_option'] = ans_opt(pubmed_l)
pubmed_u['answer_option'] = ans_opt(pubmed_u)
pubmed_a['answer_option'] = ans_opt(pubmed_a)

cols = ['context', 'question', 'answer', 'answer_option']

pubmed_l.columns, pubmed_u.columns, pubmed_a.columns = cols, cols, cols


def join_contxt(s): return s.apply(lambda x: ' '.join(x))


pubmed_l['context'] = join_contxt(pubmed_l['context'])
pubmed_u['context'] = join_contxt(pubmed_u['context'])
pubmed_a['context'] = join_contxt(pubmed_a['context'])

# pubmed_l['context'] = pubmed_l['context'].apply(lambda x: ' '.join(x))
# pubmed_u['context'] = pubmed_u['context'].apply(lambda x: ' '.join(x))
# pubmed_a['context'] = pubmed_a['context'].apply(lambda x: ' '.join(x))


def write(df, p): return df.to_csv(
    f'./dataset/PubMedQA/ori_pqa{p}.csv', index=False)


write(pubmed_l, 'l'), write(pubmed_u, 'u'), write(pubmed_a, 'a')
# pubmed_l.to_csv('./dataset/PubMedQA/ori_pqal.csv', index=False)
# pubmed_u.to_csv('./dataset/PubMedQA/ori_pqau.csv', index=False)
# pubmed_a.to_csv('./dataset/PubMedQA/ori_pqaa.csv', index=False)`


def medmcqa_process(df):
    for i, answer in enumerate(df['answer']):
        df.loc[i, 'answer'] = df.loc[i,
                                     'answer_option'][df.loc[i, 'answer'] - 1]

    return df


# MedMCQA
df_dev = pd.read_json('./dataset/MCQA/dev.json', lines=True)
df_test = pd.read_json('./dataset/MCQA/test.json', lines=True)
df_train = pd.read_json('./dataset/MCQA/train.json', lines=True)

mcqa_dev = df_dev[['subject_name', 'question']]
mcqa_dev.loc[:, 'answer_option'] = pd.Series(
    df_dev[['opa', 'opb', 'opc', 'opd']].values.tolist())
mcqa_dev.columns = ['context', 'question', 'answer_option']
mcqa_dev.loc[:, 'answer'] = df_dev.loc[:, 'cop']
medmcqa_process(mcqa_dev)

mcqa_test = df_test[['subject_name', 'question']]
mcqa_test.loc[:, 'answer_option'] = pd.Series(
    df_test[['opa', 'opb', 'opc', 'opd']].values.tolist())
mcqa_test.columns = ['context', 'question', 'answer_option']

mcqa_train = df_train[['subject_name', 'question']]
mcqa_train.loc[:, 'answer_option'] = pd.Series(
    df_train[['opa', 'opb', 'opc', 'opd']].values.tolist())
mcqa_train.columns = ['context', 'question', 'answer_option']
mcqa_train.loc[:, 'answer'] = df_train.loc[:, 'cop']
medmcqa_process(mcqa_train)

mcqa_dev.to_csv('./dataset/MCQA/dev.csv', index=False)
mcqa_test.to_csv('./dataset/MCQA/test.csv', index=False)
mcqa_train.to_csv('./dataset/MCQA/train.csv', index=False)


def process(df):
    context = 'meta_info' if 'metamap_phrases' not in df.columns else 'metamap_phrases'
    data = df[[context, 'question', 'options', 'answer']]
    data = data.rename(columns={context: 'context',
                       'question': 'question', 'options': 'answer_option'})
    if (context == 'metamap_phrases'):
        data['context'] = data['context'].apply(lambda x: ', '.join(x))
    if (type(data['answer_option'][0]) == dict):
        data['answer_option'] = data['answer_option'].apply(
            lambda x: list(x.values()))
    if (type(data['answer'][0]) == str and len(data['answer'][0]) == 1):
        for i, answer in enumerate(data['answer']):
            data['answer'][i] = data['answer_option'][i][ord(
                data['answer'][i]) - 65]
    return data


# MedQA-USMLE
dirs = read_data('./dataset/data_clean/')

file = [pd.read_json(file_path, lines=True) for file_path in dirs]

datasets = [(process(file), file_path) for file, file_path in zip(file, dirs)]

for (dataset, file_path) in datasets:
    dataset.to_csv(file_path[:-5] + 'csv', index=False)
