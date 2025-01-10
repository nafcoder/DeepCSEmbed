import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import shap
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import torch
from collections import Counter
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import pickle


def make_string(s):
    str = ''
    for i in s:
        str += i + ", "
    return str[:-2]


shap.initjs()
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

feature_paths = {
    'ProtT5-XL-U50': '/media/nafiislam/T7/DeepCSEmbed/all_features/s_linked/ProtT5-XL-U50_training_X.csv',
    'ESM-2': '/media/nafiislam/T7/DeepCSEmbed/all_features/s_linked/ESM_2_training_X.csv',
}

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
smote = SMOTE(random_state=1)
X, y = smote.fit_resample(X, y)

print(X.shape)
print(y.shape)

c = Counter(y)
print(c)


protT5_size = 1024
esm_2_size = 1280

with open('CAT_SMOTE.pkl', 'rb') as f1:
    model = pickle.load(f1)

explainer = shap.Explainer(model)
shap_values = explainer(X)

all_shap_values = np.squeeze(shap_values.values.copy())

print(all_shap_values.shape)

np.savetxt('shap_values.csv', all_shap_values, delimiter=',')

# Modify SHAP values
df = pd.DataFrame({
    'ProtT5-XL-U50': np.mean(np.abs(all_shap_values[:, :protT5_size]), axis=1)*100 ,
    'ESM-2': np.mean(np.abs(all_shap_values[:, protT5_size:]), axis=1)*100 
})

shap_values.values = df.values

shap_values.feature_names = ['ProtT5-XL-U50 (1024)', 'ESM-2 (1280)']

print("sizes")
print(shap_values.values.shape)
print(shap_values.data.shape)

shap.plots.bar(shap_values)

