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
import torch.nn.functional as F
import shap


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
    'ProtT5-XL-U50': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/ProtT5-XL-U50_independent.csv',
    'ESM-2': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/ESM_2_independent.csv',
    'Word_embedding': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/word_embedding_independent.csv'
}

# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Read training data
others = ['ESM-2', 'Word_embedding']
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

print(feature_X_Benchmark_embeddings.shape)
print(feature_y_Benchmark_embeddings.shape)

X = feature_X_Benchmark_embeddings
y = feature_y_Benchmark_embeddings

# Under-sample the dataset
rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(X, y)

c = Counter(y)
print(c)

# Inception block
class InceptionBlock(nn.Module):
    def __init__(self, filters, input_channels=21):
        super(InceptionBlock, self).__init__()
        # Path 1: 1x1 Convolution
        self.conv1x1 = nn.Conv1d(in_channels=input_channels, out_channels=filters, kernel_size=1)
        
        # Path 2: 1x1 followed by 3x3 Convolution
        self.conv1x1_3x3 = nn.Conv1d(in_channels=input_channels, out_channels=filters, kernel_size=1)
        self.conv3x3 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        
        # Path 3: 1x1 followed by 5x5 Convolution
        self.conv1x1_5x5 = nn.Conv1d(in_channels=input_channels, out_channels=filters, kernel_size=1)
        self.conv5x5 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=5, padding=2)
        
        # Path 4: MaxPooling followed by 1x1 Convolution
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = nn.Conv1d(in_channels=input_channels, out_channels=filters, kernel_size=1)
        
        # Batch normalization for faster convergence
        self.batch_norm = nn.BatchNorm1d(filters * 4)

    def forward(self, x):
        path1 = F.relu(self.conv1x1(x))
        path2 = F.relu(self.conv3x3(F.relu(self.conv1x1_3x3(x))))
        path3 = F.relu(self.conv5x5(F.relu(self.conv1x1_5x5(x))))
        path4 = F.relu(self.pool_proj(self.pool(x)))
        
        # Concatenate along the channel dimension
        out = torch.cat((path1, path2, path3, path4), dim=1)
        out = self.batch_norm(out)
        return out


# Inception-based embedding module
class InceptionEmbeddingModule(nn.Module):
    def __init__(self, input_size, embedding_dim=21):
        super(InceptionEmbeddingModule, self).__init__()
        
        # Inception blocks with progressively more filters
        self.inception1 = InceptionBlock(filters=32, input_channels=embedding_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.inception2 = InceptionBlock(filters=64, input_channels=32 * 4)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.inception3 = InceptionBlock(filters=128, input_channels=64 * 4)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4, 256)  # Dynamically set filters * 4
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Rearrange for Conv1D (batch, channels, length)

        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.pool2(x)
        x = self.inception3(x)
        x = self.pool3(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class FinalModel(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3, embedding_dim=21):
        super(FinalModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=21)
        
        # Inception blocks with progressively more filters
        self.inception1 = InceptionBlock(filters=32, input_channels=embedding_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.inception2 = InceptionBlock(filters=64, input_channels=32 * 4)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.inception3 = InceptionBlock(filters=128, input_channels=64 * 4)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.inception_fc1 = nn.Linear(128 * 4, 256)  # Dynamically set filters * 4
        self.inception_bn1 = nn.BatchNorm1d(256)
        self.inception_relu1 = nn.ReLU()
        self.inception_dropout1 = nn.Dropout(0.5)

        self.fnn_fc1 = nn.Linear(input_size-15, 500)
        self.fnn_bn1 = nn.BatchNorm1d(500)  # Batch Normalization
        self.fnn_relu1 = nn.LeakyReLU()  # Leaky ReLU
        self.fnn_dropout1 = nn.Dropout(dropout_rate)  # Dropout

        self.fnn_fc2 = nn.Linear(500, 250)
        self.fnn_bn2 = nn.BatchNorm1d(250)  # Batch Normalization
        self.fnn_relu2 = nn.LeakyReLU()  # Leaky ReLU
        self.fnn_dropout2 = nn.Dropout(dropout_rate)  # Dropout

        self.fc1 = nn.Linear(506, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(500, 250)
        self.bn2 = nn.BatchNorm1d(250)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(250, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        inception_x = self.embedding(x[:, -15:].int())
        inception_x = inception_x.permute(0, 2, 1)
        inception_x = self.inception1(inception_x)
        inception_x = self.pool1(inception_x)
        inception_x = self.inception2(inception_x)
        inception_x = self.pool2(inception_x)
        inception_x = self.inception3(inception_x)
        inception_x = self.pool3(inception_x)
        inception_x = self.flatten(inception_x)
        inception_x = self.inception_fc1(inception_x)
        inception_x = self.inception_bn1(inception_x)
        inception_x = self.inception_relu1(inception_x)
        inception_x = self.inception_dropout1(inception_x)
        
        fnn_x = x[:, :-15]
        fnn_x = self.fnn_fc1(fnn_x)
        fnn_x = self.fnn_bn1(fnn_x)
        fnn_x = self.fnn_relu1(fnn_x)
        fnn_x = self.fnn_dropout1(fnn_x)
        fnn_x = self.fnn_fc2(fnn_x)
        fnn_x = self.fnn_bn2(fnn_x)
        fnn_x = self.fnn_relu2(fnn_x)
        fnn_x = self.fnn_dropout2(fnn_x)

        x = torch.cat((inception_x, fnn_x), dim=1)

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
    

# Set parameters
input_size = X.shape[1]
batch_size = 1000

model = torch.load('Model.pth')
model.eval()

embeddings = None

# Hook for the gradients
def get_activation(name):
    def hook(model, input, output):
        global embeddings  # Refer to the outer 'embeddings' variable
        embeddings = output.detach().numpy()  # Store the output (activation) in the outer scope
    return hook

# Register the hook to the inception1 layer
model.embedding.register_forward_hook(get_activation('embedding'))

# Pass input through the model
X_tensor = torch.tensor(X).to(torch.float32)
model(X_tensor)

X = torch.tensor(embeddings).to(torch.float32)
    

# Set seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Load the trained model
model = torch.load('inception.pth')
model.eval()

# Convert X to a tensor and ensure it's on the correct device
X = torch.tensor(X, dtype=torch.float32)  # Change dtype to long if using embeddings

# Create a SHAP GradientExplainer
explainer = shap.GradientExplainer(model, X)

# Calculate SHAP values
shap_values = explainer(X)

# Extract SHAP values and squeeze them to get rid of the extra dimensions
all_shap_values = np.squeeze(shap_values.values.copy())

# Check shape of SHAP values
print(all_shap_values.shape)

# sum over the last dimension
all_shap_values = np.sum(np.abs(all_shap_values), axis=2)

print(all_shap_values.shape)
np.savetxt('shap_values.csv', all_shap_values, delimiter=',')

