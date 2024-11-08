import pandas as pd
import numpy as np
from collections import Counter
import random
import csv
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from catboost import CatBoostClassifier


def find_metrics(y_predict, y_proba, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    bal_acc = balanced_accuracy_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)

    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)

    if prec == 0 and sensitivity == 0:
        f1_score_1 = 0
    else:
        f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba)
    auPR = average_precision_score(y_test, y_proba)  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR


feature_paths = {
    'ProtT5-XL-U50': '/media/nafiislam/T7/DeepCSEmbed/all_features/s_linked/ProtT5-XL-U50_training_X.csv',
    'ESM-2': '/media/nafiislam/T7/DeepCSEmbed/all_features/s_linked/ESM_2_training_X.csv',
}

# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Read training data
others = ['ESM-2']
file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-U50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values), axis=1)

for other in others:
    file_path_Benchmark_embeddings = feature_paths[other]
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
    feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values), axis=1)

feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 0, axis=1)

X = feature_X_Benchmark_embeddings
y = feature_y_Benchmark_embeddings

print(X.shape)
print(y.shape)

# Prepare CSV writer for logging
f = open("./output_csvs/CAT_undersample.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["Type", "PREC", "F1", "AUPR", "ACC",  "SP", "MCC", "SN", "AUC", "BACC"])

# Training and evaluation loop

random.seed(0)
np.random.seed(0)

with open('CAT_undersample.pkl', 'rb') as f1:
    model = pickle.load(f1)

predicted_y = model.predict(X)
predicted_prob = model.predict_proba(X)[:, 1]

sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(predicted_y, predicted_prob, y)

writer.writerow([f'CAT_undersample',f'{prec:.3f}', f'{f1_score_1:.3f}', f'{auPR:.3f}', f'{acc:.3f}', f'{specificity:.3f}', f'{mcc:.3f}', f'{sensitivity:.3f}', f'{auc:.3f}', f'{bal_acc:.3f}'])

# Close the CSV file
f.close()