import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import shap
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import torch
from collections import Counter
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn


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
    'ProtT5-XL-U50': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/ProtT5-XL-U50_Benchmark.csv',
    'ESM-2': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/ESM_2_Benchmark.csv',
}
others = ['ESM-2']

file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-U50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)

file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-U50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values),
                                                axis=1)

feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 0, axis=1)
D_feature = None
for other in others:
    file_path_Benchmark_embeddings = feature_paths[other]
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
    feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values),
                                                    axis=1)
    D_feature = None
                                                    

X = feature_X_Benchmark_embeddings
y = feature_y_Benchmark_embeddings

D_feature = None
feature_X_Benchmark_embeddings = None
feature_y_Benchmark_embeddings = None

print(X.shape)
print(y.shape)

rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(X, y)

print(X.shape)
print(y.shape)

c = Counter(y)
print(c)

print(X.shape)
print(y.shape)


protT5_size = 1024
esm_2_size = 1280

print(X.shape)

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(BinaryClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, 500)
        self.bn1 = nn.BatchNorm1d(500)  # Batch Normalization
        self.relu1 = nn.LeakyReLU()  # Leaky ReLU
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout

        self.fc2 = nn.Linear(500, 250)
        self.bn2 = nn.BatchNorm1d(250)  # Batch Normalization
        self.relu2 = nn.LeakyReLU()  # Leaky ReLU
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout

        self.fc3 = nn.Linear(250, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

input_size = X.shape[1]

model = torch.load('undersample.pth')
model.eval() # Set the model to evaluation mode

X = torch.tensor(X, dtype=torch.float32)
explainer = shap.DeepExplainer(model, X)

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

shap_values.feature_names = ['ProtT5-XL-U50', 'ESM-2']

print(shap_values.values.shape)
print(shap_values.data.shape)

shap.plots.bar(shap_values)

