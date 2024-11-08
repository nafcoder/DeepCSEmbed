import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from collections import Counter
import random
import csv
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


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
adasyn = ADASYN(random_state=1)
X, y = adasyn.fit_resample(X, y)

c = Counter(y)
print(c)


# Define the neural network model
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


# Set parameters
input_size = X.shape[1]
batch_size = 2048

# Create DataLoader for training and testing
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X).to(torch.float32), torch.tensor(y).to(torch.float32))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(feature_X_Benchmark_embeddings_test).to(torch.float32), torch.tensor(feature_y_Benchmark_embeddings_test).to(torch.float32))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=feature_X_Benchmark_embeddings_test.shape[0], shuffle=True)

X = None
y = None
feature_X_Benchmark_embeddings_test = None
feature_y_Benchmark_embeddings_test = None

# Prepare CSV writer for logging
f = open("./output_csvs/FNN_ADASYN.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["Model", "AUPR", "AUC"])

# Training and evaluation loop

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

model = BinaryClassifier(input_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

model.train()

# Training loop
for epoch in range(500):
    for i, (inputs, labels) in enumerate(train_dataloader):
        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels.view(-1, 1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            print(f'Epoch [{epoch + 1}/{500}], Step [{i + 1}/{len(train_dataloader)}]')
    
print("Training finished.")

model.eval()

with torch.no_grad():

    for i, (inputs, labels) in enumerate(test_dataloader):
        predicted_probs = model(inputs)
        predicted_prob_numpy = predicted_probs.numpy().ravel()

        predicted_classes = (predicted_probs > 0.5).float()  # Applying a threshold of 0.5

        predicted_y_numpy = predicted_classes.numpy().ravel()

        sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(
            predicted_y_numpy, predicted_prob_numpy, labels.numpy())

        print(f'Step [{i + 1}/{len(test_dataloader)}]')

    writer.writerow([f'FNN_ADASYN',f'{auPR:.3f}', f'{auc:.3f}'])

# Close the CSV file
f.close()

