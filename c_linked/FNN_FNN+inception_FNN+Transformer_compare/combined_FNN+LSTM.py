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
    'ProtT5-XL-U50': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/ProtT5-XL-U50_training.csv',
    'ESM-2': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/ESM_2_training.csv',
    'Word_embedding': '/media/nafiislam/T7/DeepCSEmbed/all_features/c_linked/word_embedding_training.csv',
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
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, :].values), axis=1)

for other in others:
    file_path_Benchmark_embeddings = feature_paths[other]
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
    feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values), axis=1)

feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 0, axis=1)

feature_X_Benchmark_embeddings_positive = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 1,
                                          1:]
feature_y_Benchmark_embeddings_positive = feature_X_Benchmark_embeddings[
    feature_X_Benchmark_embeddings[:, 0] == 1, 0].astype('int')

feature_X_Benchmark_embeddings_negative = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 0,
                                          1:]
feature_y_Benchmark_embeddings_negative = feature_X_Benchmark_embeddings[
    feature_X_Benchmark_embeddings[:, 0] == 0, 0].astype('int')

D_feature = None
feature_X_Benchmark_embeddings = None

print(feature_X_Benchmark_embeddings_positive.shape)
print(feature_y_Benchmark_embeddings_positive.shape)

print(feature_X_Benchmark_embeddings_negative.shape)
print(feature_y_Benchmark_embeddings_negative.shape)

feature_X_Benchmark_embeddings_positive_train, feature_X_Benchmark_embeddings_positive_test, feature_y_Benchmark_embeddings_positive_train, feature_y_Benchmark_embeddings_positive_test = train_test_split(feature_X_Benchmark_embeddings_positive, feature_y_Benchmark_embeddings_positive, test_size=24, random_state=1)
feature_X_Benchmark_embeddings_negative_train, feature_X_Benchmark_embeddings_negative_test, feature_y_Benchmark_embeddings_negative_train, feature_y_Benchmark_embeddings_negative_test = train_test_split(feature_X_Benchmark_embeddings_negative, feature_y_Benchmark_embeddings_negative, test_size=52, random_state=1)

feature_X_Benchmark_embeddings_positive = None
feature_y_Benchmark_embeddings_positive = None
feature_X_Benchmark_embeddings_negative = None
feature_y_Benchmark_embeddings_negative = None

feature_X_Benchmark_embeddings_train = np.concatenate(
    (feature_X_Benchmark_embeddings_positive_train, feature_X_Benchmark_embeddings_negative_train), axis=0)
feature_y_Benchmark_embeddings_train = np.concatenate(
    (feature_y_Benchmark_embeddings_positive_train, feature_y_Benchmark_embeddings_negative_train), axis=0)
feature_X_Benchmark_embeddings_test = np.concatenate(
    (feature_X_Benchmark_embeddings_positive_test, feature_X_Benchmark_embeddings_negative_test), axis=0)
feature_y_Benchmark_embeddings_test = np.concatenate(
    (feature_y_Benchmark_embeddings_positive_test, feature_y_Benchmark_embeddings_negative_test), axis=0)

feature_X_Benchmark_embeddings_positive_train = None
feature_y_Benchmark_embeddings_positive_train = None
feature_X_Benchmark_embeddings_negative_train = None
feature_y_Benchmark_embeddings_negative_train = None
feature_X_Benchmark_embeddings_positive_test = None
feature_y_Benchmark_embeddings_positive_test = None
feature_X_Benchmark_embeddings_negative_test = None
feature_y_Benchmark_embeddings_negative_test = None

print(feature_X_Benchmark_embeddings_train.shape)
print(feature_y_Benchmark_embeddings_train.shape)

X = feature_X_Benchmark_embeddings_train
y = feature_y_Benchmark_embeddings_train

print(feature_X_Benchmark_embeddings_test.shape)
print(feature_y_Benchmark_embeddings_test.shape)

# Under-sample the dataset
rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(X, y)

c = Counter(y)
print(c)


# RNN Encoder block
class RNNBlock(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=2, dropout=0.1):
        super(RNNBlock, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        # Expecting input shape: (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)
        return out

class FinalModel(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3, embedding_dim=21):
        super(FinalModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        
        # RNN branch for sequence data
        self.rnn_block = RNNBlock(input_size=embedding_dim, hidden_size=64, num_layers=2, dropout=0.1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Fully connected layers after RNN
        self.rnn_fc1 = nn.Linear(15 * 32, 256)
        self.rnn_bn1 = nn.BatchNorm1d(256)
        self.rnn_relu1 = nn.ReLU()
        self.rnn_dropout1 = nn.Dropout(0.5)

        # Fully connected layers for the FNN branch
        self.fnn_fc1 = nn.Linear(input_size-15, 500)
        self.fnn_bn1 = nn.BatchNorm1d(500)  # Batch Normalization
        self.fnn_relu1 = nn.LeakyReLU()  # Leaky ReLU
        self.fnn_dropout1 = nn.Dropout(dropout_rate)  # Dropout

        self.fnn_fc2 = nn.Linear(500, 250)
        self.fnn_bn2 = nn.BatchNorm1d(250)  # Batch Normalization
        self.fnn_relu2 = nn.LeakyReLU()  # Leaky ReLU
        self.fnn_dropout2 = nn.Dropout(dropout_rate)  # Dropout

        # Final fully connected layers
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
        # RNN branch
        rnn_x = self.embedding(x[:, -15:].int())
        rnn_x = self.rnn_block(rnn_x)
        rnn_x = self.pool(rnn_x)
        rnn_x = self.flatten(rnn_x)
        rnn_x = self.rnn_fc1(rnn_x)
        rnn_x = self.rnn_bn1(rnn_x)
        rnn_x = self.rnn_relu1(rnn_x)
        rnn_x = self.rnn_dropout1(rnn_x)
        
        # FNN branch
        fnn_x = x[:, :-15]
        fnn_x = self.fnn_fc1(fnn_x)
        fnn_x = self.fnn_bn1(fnn_x)
        fnn_x = self.fnn_relu1(fnn_x)
        fnn_x = self.fnn_dropout1(fnn_x)
        fnn_x = self.fnn_fc2(fnn_x)
        fnn_x = self.fnn_bn2(fnn_x)
        fnn_x = self.fnn_relu2(fnn_x)
        fnn_x = self.fnn_dropout2(fnn_x)

        # Concatenate both RNN and FNN branches
        x = torch.cat((rnn_x, fnn_x), dim=1)

        # Final fully connected layers
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

# Create DataLoader for training and testing
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X).to(torch.float32), torch.tensor(y).to(torch.float32))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X = None
y = None

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

model =  FinalModel(input_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.train()

# Training loop
for epoch in range(50):
    for i, (inputs, labels) in enumerate(train_dataloader):
        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels.view(-1, 1))

        # Backward pass and optimization
        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Print gradient norm
        # print("Gradient norm:", gradient_norm)

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            print(f'Epoch [{epoch + 1}/{50}], Step [{i + 1}/{len(train_dataloader)}]')


print("Training finished.")

file = f'Model_LSTM.pth'
torch.save(model, file)

X_test = feature_X_Benchmark_embeddings_test
y_test = feature_y_Benchmark_embeddings_test


model = torch.load('Model_LSTM.pth')
model.eval()

with torch.no_grad():
    outputs = model(torch.tensor(X_test).to(torch.float32))
    y_predict = (outputs > 0.5).int().numpy().flatten()
    y_proba = outputs.numpy().flatten()

sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(y_predict, y_proba, y_test)

print("AUC:", round(auc, 3))
print("auPR:", round(auPR, 3))