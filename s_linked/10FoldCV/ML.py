from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
import numpy as np
import random
import csv
from sklearn.metrics import roc_curve
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve


def find_metrics(model_name, y_test):
    if model_name == 'RF':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'MLP':
        model = MLPClassifier(random_state=1, max_iter=4000)
    elif model_name == 'SVM':
        model = SVC(kernel='rbf', random_state=1, probability=True)
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'XGB':
        model = XGBClassifier(random_state=1)
    elif model_name == 'LBM':
        model = lgb.LGBMClassifier(random_state=1)
    elif model_name == 'CAT':
        model = CatBoostClassifier(random_state=1, verbose=0)
    elif model_name == 'ADA':
        model = AdaBoostClassifier(random_state=1)
    elif model_name == 'GBC':
        model = GradientBoostingClassifier(random_state=1)
    elif model_name == 'GPC':
        model = GaussianProcessClassifier(random_state=1)
    elif model_name == 'QDA':
        model = QuadraticDiscriminantAnalysis()
    else:
        print('Wrong model name')
        return
    
    model.fit(X_train, y_train)
    
    y_predict = model.predict(X_test)  # predicted labels
    y_proba = model.predict_proba(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    bal_acc = balanced_accuracy_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)
    prec = tp / (tp + fp)
    f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    auPR = average_precision_score(y_test, y_proba[:, 1])  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR, y_proba[:, 1]


features = {
    'ProtT5-XL-U50': '/media/nafiislam/T7/DeepCSEmbed/all_features/s_linked/ProtT5-XL-U50_training_Y.csv',
    'ESM-2': '/media/nafiislam/T7/DeepCSEmbed/all_features/s_linked/ESM_2_training_Y.csv',
}

all_model_name = ['SVM', 'RF', 'KNN', 'XGB', 'LBM', 'ADA', 'GBC', 'GPC', 'QDA', 'CAT']

# Read training data
others = ['ESM-2']
file_path_Benchmark_embeddings = features['ProtT5-XL-U50']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=True)
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 1:].values), axis=1)

for other in others:
    file_path_Benchmark_embeddings = features[other]
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

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

with open('./ML.csv', 'w') as f1:
    csvWriter = csv.writer(f1)

    csvWriter.writerow(['Classifier', "AUPR", "PREC", "F1", "SP", "SN", "BACC", "MCC", "ACC", "AUC"])
    for model_name in all_model_name:
        print(model_name)
        random.seed(1)
        
        local_Sensitivity = []
        local_Specificity = []
        local_Balanced_acc = []
        local_Accuracy = []
        local_Precision = []
        local_AUPR = []
        local_F1 = []
        local_MCC = []
        local_AUC = []

        global_y = []
        global_y_proba = []

        i = 1
        for train_index, test_index in cv.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR, y_proba = find_metrics(model_name, y_test)

            local_Sensitivity.append(sensitivity)
            local_Specificity.append(specificity)
            local_Balanced_acc.append(bal_acc)
            local_Accuracy.append(acc)
            local_Precision.append(prec)
            local_F1.append(f1_score_1)
            local_MCC.append(mcc)
            local_AUC.append(auc)
            local_AUPR.append(auPR)

            global_y.extend(y_test)
            global_y_proba.extend(y_proba)

            print(i, 'th iteration done')
            i = i + 1
            print('___________________________________________________________________________________________________________')

        print('classifier : ', model_name)
        print('Sensitivity : {0:.3f}'.format(np.mean(local_Sensitivity)))
        print('Specificity : {0:.3f}'.format(np.mean(local_Specificity)))
        print('Balanced_acc : {0:.3f}'.format(np.mean(local_Balanced_acc)))
        print('Accuracy : {0:.3f}'.format(np.mean(local_Accuracy)))
        print('Precision : {0:.3f}'.format(np.mean(local_Precision)))
        print('F1-score: {0:.3f}'.format(np.mean(local_F1)))
        print('MCC: {0:.3f}'.format(np.mean(local_MCC)))
        print('AUC: {0:.3f}'.format(np.mean(local_AUC)))
        print('auPR: {0:.3f}'.format(np.mean(local_AUPR)))

        fprs, tprs, _ = roc_curve(global_y, global_y_proba)

        f_name = f'./ROC/fpr_{model_name}.csv'

        with open(f_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[fp] for fp in fprs])
        
        f_name = f'./ROC/tpr_{model_name}.csv'
        with open(f_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[tp] for tp in tprs])

        precision, recall, _ = precision_recall_curve(global_y, global_y_proba)

        f_name = f'./precision_recall_curve/precision_{model_name}.csv'
        with open(f_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[p] for p in precision])
        
        f_name = f'./precision_recall_curve/recall_{model_name}.csv'
        with open(f_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[r] for r in recall])

        mean_sensitivity = np.mean(local_Sensitivity)
        std_sensitive = np.std(local_Sensitivity)
        mean_specificity = np.mean(local_Specificity)
        std_specificity = np.std(local_Specificity)
        mean_bal_acc = np.mean(local_Balanced_acc)
        std_bal_acc = np.std(local_Balanced_acc)
        mean_acc = np.mean(local_Accuracy)
        std_acc = np.std(local_Accuracy)
        mean_prec = np.mean(local_Precision)
        std_prec = np.std(local_Precision)
        mean_f1_score_1 = np.mean(local_F1)
        std_f1_score_1 = np.std(local_F1)
        mean_mcc = np.mean(local_MCC)
        std_mcc = np.std(local_MCC)
        mean_auc = np.mean(local_AUC)
        std_auc = np.std(local_AUC)
        mean_auPR = np.mean(local_AUPR)
        std_auPR = np.std(local_AUPR)

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

        csvWriter.writerow([model_name] + row)

        print('___________________________________________________________________________________________________________')
        print('___________________________________________________________________________________________________________')
        print('___________________________________________________________________________________________________________')

        f1.flush()
