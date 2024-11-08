from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import random
import csv
from sklearn.preprocessing import MinMaxScaler


def find_metrics(y_predict, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    bal_acc = balanced_accuracy_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)
    prec = tp / (tp + fp)
    f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc


with open('/media/nafiislam/T7/C_linked_Glycosylation/dataset/GlycoEP_pred.fasta', 'r') as f:
    lines = f.readlines()

    pred_target = []

    for i in range(0, len(lines), 2):
        splitter = lines[i].rstrip()[1:].split(',')

        for j in range(2, len(splitter), 2):
            pred_target.append(splitter[j])

print(pred_target)

with open('/media/nafiislam/T7/C_linked_Glycosylation/dataset/Benchmark.fasta', 'r') as f:
    lines = f.readlines()

    target = []

    for i in range(0, len(lines), 2):
        splitter = lines[i].rstrip()[1:].split(',')

        for j in range(2, len(splitter), 2):
            target.append(splitter[j])

print(target)

sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc = find_metrics(pred_target, target)

print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Balanced Accuracy: {bal_acc}')
print(f'Accuracy: {acc}')
print(f'Precision: {prec}')
print(f'F1 Score: {f1_score_1}')
print(f'MCC: {mcc}')