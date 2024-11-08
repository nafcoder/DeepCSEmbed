import pandas as pd
import numpy as np
from collections import Counter
import random
import csv
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader


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


with open('predicted_values.txt', 'w') as f:

    D_feature = pd.read_csv('features.csv', header=None, low_memory=False)
    feature_X_Benchmark = D_feature.values
    print('feature_X_Benchmark : ', feature_X_Benchmark.shape)

    X = feature_X_Benchmark

    print('X : ', X.shape)

    # Define the neural network model

    model = torch.load('undersample.pth')
    model.eval()

    # Predict the test data
    X_test = torch.tensor(X, dtype=torch.float32)

    y_proba = model(X_test).detach().numpy()
    y_predict = (y_proba > 0.5).astype(int)

    for i in range(len(y_predict)):
        f.write(str(y_predict[i][0]) + ',' + str(y_proba[i][0]) + '\n')
    
    print('Predictions written to predicted_values.txt')

    
