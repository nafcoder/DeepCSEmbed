from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
import numpy as np
import csv
import pickle
import random
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import math

def get_top_indices(arr, indices):
    top_indices = np.argsort(arr)[:indices]
    
    array = np.zeros(len(arr), dtype=bool)

    array[top_indices] = True
    
    return array

rankings = array = np.loadtxt('rfe.csv', delimiter=',')

others = ['ESM-2', 'PSSM', 'Monogram', 'DPC', 'ASA', 'HSE', 'TA']

prott5_size = 1024
esm2_size = 1280
pssm_size = 20
monogram_size = 1
dpc_size = 400
asa_size = 1
hse_size = 2
ta_size = 2


prott5_count = 0
esm2_count = 0
pssm_count = 0
monogram_count = 0
dpc_count = 0
asa_count = 0
hse_count = 0
ta_count = 0

support = get_top_indices(rankings, 100)
for i in range(len(support)):
    if support[i] == True:
        if i < prott5_size:
            prott5_count += 1
        elif i < prott5_size + esm2_size:
            esm2_count += 1
        elif i < prott5_size + esm2_size + pssm_size:
            pssm_count += 1
        elif i < prott5_size + esm2_size + pssm_size + monogram_size:
            monogram_count += 1
        elif i < prott5_size + esm2_size + pssm_size + monogram_size + dpc_size:
            dpc_count += 1
        elif i < prott5_size + esm2_size + pssm_size + monogram_size + dpc_size + asa_size:
            asa_count += 1
        elif i < prott5_size + esm2_size + pssm_size + monogram_size + dpc_size + asa_size + hse_size:
            hse_count += 1
        else:
            ta_count += 1

prott5_contrib = prott5_count * 1 / math.sqrt(prott5_size)
esm2_contrib = esm2_count * 1 / math.sqrt(esm2_size)
pssm_contrib = pssm_count * 1 / math.sqrt(pssm_size)
monogram_contrib = monogram_count * 1 / math.sqrt(monogram_size)
dpc_contrib = dpc_count * 1 / math.sqrt(dpc_size)
asa_contrib = asa_count * 1 / math.sqrt(asa_size)
hse_contrib = hse_count * 1 / math.sqrt(hse_size)
ta_contrib = ta_count * 1 / math.sqrt(ta_size)

values = [prott5_contrib, esm2_contrib, pssm_contrib, monogram_contrib, dpc_contrib, asa_contrib, hse_contrib, ta_contrib]

labels = ['Prott5', 'ESM2', 'PSSM', 'Monogram', 'DPC', 'ASA', 'HSE', 'TA']

plt.figure(figsize=(6, 4))
plt.bar(labels, values, color='skyblue')

plt.title('Feature Selection by RFE (XGB)')
plt.xlabel('Feature groups')
plt.ylabel('Contribution among the top 100 ranked')

plt.show()

