import numpy as np
from pyswarm import pso
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import random
import csv
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

total = 0
# Define the fitness function for PSO
def fitness_function(features):
    global total
    # Convert binary array to integer indices of selected features
    selected_features = np.where(features > 0.5)[0]

    if len(selected_features) == 0:
        return 1e6  # Penalize if no features are selected

    # Train a SVM model on the selected features
    model = SVC(kernel='rbf', random_state=1, probability=True)
    scores = cross_val_score(model, X[:, selected_features], y, cv=5, scoring='accuracy')

    total += 1
    
    print(total)

    # Return negative mean accuracy as PSO minimizes the fitness function
    return -np.mean(scores)


random.seed(0)
np.random.seed(0)

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

lb = [0] * X.shape[1]  # Lower bounds of the features (0 means not selected)
ub = [1] * X.shape[1]  # Upper bounds of the features (1 means selected)
best_features, _ = pso(fitness_function, lb, ub, swarmsize=30, maxiter=90)

f = open("./pso.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[fe] for fe in best_features])
f.close()