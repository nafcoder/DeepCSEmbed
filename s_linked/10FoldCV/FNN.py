import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import random
import csv
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


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


# Define file paths
feature_paths = {
    'ProtT5-XL-U50': '/media/nafiislam/T7/DeepCSEmbed/all_features/s_linked/ProtT5-XL-U50_training_Y.csv',
    'ESM-2': '/media/nafiislam/T7/DeepCSEmbed/all_features/s_linked/ESM_2_training_Y.csv',
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

feature_X_Benchmark_embeddings_train = feature_X_Benchmark_embeddings
feature_y_Benchmark_embeddings_train = feature_y_Benchmark_embeddings

X = feature_X_Benchmark_embeddings_train
y = feature_y_Benchmark_embeddings_train

print(feature_X_Benchmark_embeddings_train.shape)
print(feature_y_Benchmark_embeddings_train.shape)

D_feature = None
feature_X_Benchmark_embeddings_train = None
feature_y_Benchmark_embeddings_train = None
feature_X_Benchmark_embeddings = None
feature_y_Benchmark_embeddings = None

# Under-sample the dataset
rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(X, y)

c = Counter(y)
print(c)

# Define the neural network model
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
epochs = [500]

# Prepare CSV writer for logging
f = open("./FNN.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(['Classifier', "AUPR", "PREC", "F1", "SP", "SN", "BACC", "MCC", "ACC", "AUC"])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

test_sensitivity = []
test_specificity = []
test_balanced_accuracy = []
test_accuracy = []
test_precision = []
test_f1 = []
test_mcc = []
test_auc = []
test_aupr = []

global_y = []
global_y_proba = []

itr = 0
for train_index, test_index in cv.split(X, y):
    itr += 1
    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]
    # Create DataLoader for training and testing
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).to(torch.float32), torch.tensor(y_train).to(torch.float32))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).to(torch.float32), torch.tensor(y_test).to(torch.float32))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=X_test.shape[0], shuffle=True)
    
    # Training and evaluation loop

    for num_epoch in epochs:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        model = BinaryClassifier(input_size)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        model.train()

        # Training loop
        for epoch in range(num_epoch):
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
                    print(f'Epoch [{epoch + 1}/{num_epoch}], Step [{i + 1}/{len(train_dataloader)}], iteration {itr}')

        print("Training finished.")

        model.eval()

        # Test the model: we don't need to compute gradients
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
                predicted_probs = model(inputs)
                predicted_prob_numpy = predicted_probs.numpy().ravel()

                predicted_classes = (predicted_probs > 0.5).float()  # Applying a threshold of 0.5

                predicted_y_numpy = predicted_classes.numpy().ravel()

                sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(
                    predicted_y_numpy, predicted_prob_numpy, labels.numpy())

                test_sensitivity.append(sensitivity)
                test_specificity.append(specificity)
                test_balanced_accuracy.append(bal_acc)
                test_accuracy.append(acc)
                test_precision.append(prec)
                test_f1.append(f1_score_1)
                test_mcc.append(mcc)
                test_auc.append(auc)
                test_aupr.append(auPR)

                global_y.extend(labels.numpy().tolist())
                global_y_proba.extend(predicted_prob_numpy.tolist())

                # Compute loss
                loss = criterion(predicted_probs, labels.view(-1, 1))

                print(f'Step [{i + 1}/{len(test_dataloader)}]')

mean_sensitivity = np.mean(test_sensitivity)
std_sensitive = np.std(test_sensitivity)
mean_specificity = np.mean(test_specificity)
std_specificity = np.std(test_specificity)
mean_bal_acc = np.mean(test_balanced_accuracy)
std_bal_acc = np.std(test_balanced_accuracy)
mean_acc = np.mean(test_accuracy)
std_acc = np.std(test_accuracy)
mean_prec = np.mean(test_precision)
std_prec = np.std(test_precision)
mean_f1_score_1 = np.mean(test_f1)
std_f1_score_1 = np.std(test_f1)
mean_mcc = np.mean(test_mcc)
std_mcc = np.std(test_mcc)
mean_auc = np.mean(test_auc)
std_auc = np.std(test_auc)
mean_auPR = np.mean(test_aupr)
std_auPR = np.std(test_aupr)

mean_std_format = lambda mean, std: f'{mean:.3f} ± {std:.3f}'

row = [
    mean_std_format(mean_auPR, std_auPR),
    mean_std_format(mean_prec, std_prec),
    mean_std_format(mean_f1_score_1, std_f1_score_1),
    mean_std_format(mean_specificity, std_specificity),
    mean_std_format(mean_sensitivity, std_sensitive),
    mean_std_format(mean_bal_acc, std_bal_acc),
    mean_std_format(mean_mcc, std_mcc),
    mean_std_format(mean_acc, std_acc),
    mean_std_format(mean_auc, std_auc)
]

writer.writerow(['FNN'] + row)
f.close()

fprs, tprs, _ = roc_curve(global_y, global_y_proba)

# Save ROC curve data
f = open("./ROC/fpr_FNN.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[fp] for fp in fprs])
f.close()

f = open("./ROC/tpr_FNN.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[tp] for tp in tprs])
f.close()

precision, recall, _ = precision_recall_curve(global_y, global_y_proba)

f_name = f'./precision_recall_curve/precision_FNN.csv'
with open(f_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[p] for p in precision])

f_name = f'./precision_recall_curve/recall_FNN.csv'
with open(f_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[r] for r in recall])