import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data preprocessing')

DATASETS = {
    'pima': {
        'file': 'pima_preprocessed.csv',
        'label': '低维Pima (d=9)',
        'ndim': 9,
    },
    'creditcard': {
        'file': 'creditcard_preprocessed.csv',
        'label': '中维CreditCard (d=30)',
        'ndim': 30,
    },
    'tcga': {
        'file': 'tcga_preprocessed.csv',
        'label': '高维TCGA (d=61)',
        'ndim': 61,
    },
}


def load_dataset(name):
    info = DATASETS[name]
    path = os.path.join(DATA_DIR, info['file'])
    df = pd.read_csv(path)
    y = df['y'].values.astype(np.int64)
    X = df.iloc[:, 1:].values.astype(np.float64)
    return X, y, info


def get_param_names(name):
    info = DATASETS[name]
    path = os.path.join(DATA_DIR, info['file'])
    df = pd.read_csv(path, nrows=1)
    param_names = list(df.columns[1:])
    return param_names
