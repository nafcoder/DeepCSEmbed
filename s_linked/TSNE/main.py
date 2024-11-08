import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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

# Under-sample the dataset
rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(X, y)

c = Counter(y)
print(c)

random.seed(0)
np.random.seed(0)

with open('CAT_SMOTE.pkl', 'rb') as f:
    model = pickle.load(f)

y_predict_proba = model.predict_proba(X)

tsne_before = TSNE(n_components=2, random_state=1, perplexity=5)
tsne_before = tsne_before.fit_transform(X)

tsne_after = TSNE(n_components=2, random_state=1, perplexity=5)
tsne_after = tsne_after.fit_transform(y_predict_proba)

# plot

plt.figure(figsize=(6, 4))

plt.scatter(tsne_before[:, 0], tsne_before[:, 1], c=y, cmap='coolwarm')
plt.title("t-SNE visualization of initial feature space")
plt.xlabel("comp-1")
plt.ylabel("comp-2")
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Label 0'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Label 1')]

plt.legend(handles=handles, title="Labels", loc='best')

plt.show()

plt.figure(figsize=(6, 4))

plt.scatter(tsne_after[:, 0], tsne_after[:, 1], c=y, cmap='coolwarm')
plt.title("t-SNE visualization of model output feature space")
plt.xlabel("comp-1")
plt.ylabel("comp-2")
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Label 0'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Label 1')]

plt.legend(handles=handles, title="Labels", loc='best')

plt.show()
