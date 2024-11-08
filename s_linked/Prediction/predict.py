import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
import random
import csv
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from catboost import CatBoostClassifier


with open('predicted_values.txt', 'w') as f:

    D_feature = pd.read_csv('features.csv', header=None, low_memory=False)
    feature_X_Benchmark = D_feature.values
    print('feature_X_Benchmark : ', feature_X_Benchmark.shape)

    X = feature_X_Benchmark

    print('X : ', X.shape)

    with open('CAT_SMOTE.pkl', 'rb') as f1:
        model = pickle.load(f1)

    y_predict = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    for i in range(len(y_predict)):
        f.write(str(y_predict[i]) + ',' + str(y_proba[i]) + '\n')
    
    print('Predictions written to predicted_values.txt')

    
