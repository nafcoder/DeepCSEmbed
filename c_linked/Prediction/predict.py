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
import torch.nn.functional as F


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


with open('predicted_values.txt', 'w') as f:

    D_feature = pd.read_csv('features.csv', header=None, low_memory=False)
    feature_X_Benchmark = D_feature.values
    print('feature_X_Benchmark : ', feature_X_Benchmark.shape)

    X = feature_X_Benchmark

    print('X : ', X.shape)

    # Define the neural network model

    model = torch.load('Model.pth')
    model.eval()

    # Predict the test data
    X_test = torch.tensor(X, dtype=torch.float32)

    y_proba = model(X_test).detach().numpy()
    y_predict = (y_proba > 0.5).astype(int)

    for i in range(len(y_predict)):
        f.write(str(y_predict[i][0]) + ',' + str(y_proba[i][0]) + '\n')
    
    print('Predictions written to predicted_values.txt')

    
