from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
import numpy as np
import csv
import pickle
import random
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def preprocess_the_dataset(feature_X):
    feature_X = MinMaxScaler().fit_transform(feature_X)
    return feature_X


feature_paths = {
    'ProtT5-XL-U50': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/ProtT5-XL-U50_training.csv',
    'ESM-2': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/ESM_2_training.csv',
    'PSSM': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/pssm_training.csv', 
    'Monogram': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/monogram_training.csv',
    'DPC': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/dpc_training.csv',
    'ASA': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/asa_training.csv',
    'HSE': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/hse_training.csv',
    'TA': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/ta_training.csv',
}

# Read training data
others = ['ESM-2', 'PSSM', 'Monogram', 'DPC', 'ASA', 'HSE', 'TA']

file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-U50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values), axis=1)

for other in others:
    file_path_Benchmark_embeddings = feature_paths[other]
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
    if other == 'ESM-2':
        feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values), axis=1)
    else:
        feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, preprocess_the_dataset(D_feature.iloc[:, 1:].values)), axis=1)

feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 0, axis=1)

feature_X_Benchmark_embeddings_train = feature_X_Benchmark_embeddings
feature_y_Benchmark_embeddings_train = feature_y_Benchmark_embeddings

X = feature_X_Benchmark_embeddings_train
y = feature_y_Benchmark_embeddings_train

D_feature = None
feature_X_Benchmark_embeddings = None
feature_y_Benchmark_embeddings = None
feature_X_Benchmark_embeddings_train = None
feature_y_Benchmark_embeddings_train = None

# balance the dataset :
rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(X, y)

print(X.shape)
print(y.shape)

c = Counter(y)
print(c)

model = XGBClassifier(random_state=1)

rfe = RFE(estimator=model, n_features_to_select=100, step=0.1)

rfe.fit(X, y)

rankings = rfe.ranking_

f = open("./rfe.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[rfe] for rfe in rankings])
f.close()