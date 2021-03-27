#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:00:19 2021

@authors: Chiara Marzi - Post-doctoral fellow
         Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
         University of Bologna, Bologna, Italy.
         Email address: chiara.marzi3@unibo.it

         Stefano Diciotti - Associate Professor in Biomedical Engineering,
         Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
         University of Bologna, Bologna, Italy.
         Email address: stefano.diciotti@unibo.it
"""

import matplotlib
matplotlib.use('Agg') # To save figures in backend mode
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split, cross_validate, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import ColumnSelector
from sklearn.metrics import make_scorer, recall_score, roc_curve, auc, roc_auc_score
import time
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
#import sys
import argparse

# ignore all future warnings
simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=FutureWarning)
simplefilter("ignore", category=ConvergenceWarning)

start_time = time.time()

if not os.path.exists('Figures'):
    os.makedirs('Figures')

if not os.path.exists('csv_results'):
        os.makedirs('csv_results')

### DATA LOADING ###
parser = argparse.ArgumentParser(description='A machine learning predicitive model, optimized in a nested stratified cross-validation loop repeated 1000 times')
parser.add_argument('XLSX_file', metavar='XLSX_file', type=str,
                    help='XLSX file including data')
args = parser.parse_args()

#XLSX_file = sys.argv[1]
df = pd.read_excel(args.XLSX_file)

X = np.asarray(df.iloc[:,1:]) # predictors
y = np.asarray(df.iloc[:,0]) # target
##########################################

### VALIDATION SCHEME and PIPELINE ###
NUM_TRIALS = 1 #1000 # Repetitions of the nested stratified CV

NUM_SPLITS = 2 #10 # Nested CV splits (for the inner and outer loops)

scaler = StandardScaler(with_mean=True, with_std=True)

feature_selector = ColumnSelector()

comb = [(3, 7, 9, 11),                          # Right Hemisphere: Parieto-Occipital activity + Intrahemispheric Connectivity
        (2, 6, 8, 10),                          # Left Hemisphere: Parieto-Occipital activity + Intrahemispheric Connectivity
        (2, 3, 6, 7, 8, 9, 10, 11),             # Left + Right Hemisphere: Parieto-Occipital activity + Intrahemispheric Connectivity
        (1, 5),                                 # Right hemisphere:Frontal activity
        (0, 4),                                 # Left hemisphere:Frontal activity
        (0, 1, 4, 5),                           # Left + Right Hemisphere: Frontal activity
        (1, 3, 5, 7, 9, 11),                    # Right Hemisphere activity
        (0, 2, 4, 6, 8, 10),                    # Left Hemisphere activity
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] # Left + Right Hemisphere activity

mydict = {'(3, 7, 9, 11)': 0,                           # Right Hemisphere: Parieto-Occipital activity + Intrahemispheric Connectivity
          '(2, 6, 8, 10)': 0,                           # Left Hemisphere: Parieto-Occipital activity + Intrahemispheric Connectivity
          '(2, 3, 6, 7, 8, 9, 10, 11)': 0,              # Left + Right Hemisphere: Parieto-Occipital activity + Intrahemispheric Connectivity
          '(1, 5)': 0,                                  # Right hemisphere:Frontal activity
          '(0, 4)': 0,                                  # Left hemisphere:Frontal activity
          '(0, 1, 4, 5)': 0,                            # Left + Right Hemisphere: Frontal activity
          '(1, 3, 5, 7, 9, 11)': 0,                     # Right Hemisphere activity
          '(0, 2, 4, 6, 8, 10)': 0,                     # Left Hemisphere activity
          '(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)': 0}  # Left + Right Hemisphere activity


classifier = SVC(probability=True, kernel='linear', gamma='scale')

pipe = Pipeline(steps=[('scaler', scaler), ('feature_selector', feature_selector), ('classifier', classifier)])

p_grid = [
    {
     'feature_selector__cols': comb,
     'classifier': [SVC(probability=True, kernel='linear', gamma='scale', class_weight='balanced', max_iter=1000)],
     'classifier__C': [0.1, 0.2, 0.3]
     },

    {
     'feature_selector__cols': comb,
     'classifier': [LogisticRegression(tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=False, solver='liblinear', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=1, l1_ratio=None, penalty='l2')],
     'classifier__C': [0.1, 0.2, 0.3]
     }
    ]
##########################################

### PERFORMANCE SCORES DEFINITION ###
myscoring = {'bal_acc': 'balanced_accuracy',
             'roc_auc': 'roc_auc',
             'ave_pre': 'average_precision',
             'acc': 'accuracy',
             'sensitivity': 'recall',
             'specificity': make_scorer(recall_score,pos_label=0)
             }

acc_train_scores = np.zeros(NUM_TRIALS)
bal_acc_train_scores = np.zeros(NUM_TRIALS)
roc_auc_train_scores = np.zeros(NUM_TRIALS)
sensitivity_train_scores = np.zeros(NUM_TRIALS)
specificity_train_scores = np.zeros(NUM_TRIALS)

acc_test_scores = np.zeros(NUM_TRIALS)
bal_acc_test_scores = np.zeros(NUM_TRIALS)
roc_auc_test_scores = np.zeros(NUM_TRIALS)
sensitivity_test_scores = np.zeros(NUM_TRIALS)
specificity_test_scores = np.zeros(NUM_TRIALS)

tprs = []
mean_fpr = np.linspace(0, 1, 1000) # the average ROC curve plot will be built on 1000 points
##########################################

### TRAINING, VALIDATION and TEST ###
for i in range(NUM_TRIALS):
    print("Iteration:", i)

    outer_cv = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=i)
    inner_cv = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=i)

    clf = GridSearchCV(pipe, param_grid=p_grid, cv=inner_cv, refit='bal_acc', scoring=myscoring, n_jobs=1)
    nested_score = cross_validate(clf, X=X, y=y, cv=outer_cv, return_train_score=True, return_estimator=True, scoring = myscoring, n_jobs=1)

    acc_train_scores[i] = np.mean(nested_score['train_acc'])
    bal_acc_train_scores[i] = np.mean(nested_score['train_bal_acc'])
    roc_auc_train_scores[i] = np.mean(nested_score['train_roc_auc'])
    sensitivity_train_scores[i] = np.mean(nested_score['train_sensitivity'])
    specificity_train_scores[i] = np.mean(nested_score['train_specificity'])
    #print ('Train: acc', acc_train_scores[i])
    #print ('Train: bal_acc', bal_acc_train_scores[i])
    #print ('Train: roc_auc', roc_auc_train_scores[i])
    #print ('Train: sensitivity', sensitivity_train_scores[i])
    #print ('Train: specificity', specificity_train_scores[i])

    acc_test_scores[i] = np.mean(nested_score['test_acc'])
    bal_acc_test_scores[i] = np.mean(nested_score['test_bal_acc'])
    roc_auc_test_scores[i] = np.mean(nested_score['test_roc_auc'])
    sensitivity_test_scores[i] = np.mean(nested_score['test_sensitivity'])
    specificity_test_scores[i] = np.mean(nested_score['test_specificity'])
    #print ('Test: acc', acc_test_scores[i])
    #print ('Test: bal_acc', bal_acc_test_scores[i])
    #print ('Test: roc_auc', roc_auc_test_scores[i])
    #print ('Test: sensitivity', sensitivity_test_scores[i])
    #print ('Test: specificity', specificity_test_scores[i])

    split_iter = 0
    for train_index, test_index in outer_cv.split(X, y):
        print("Split:", split_iter)

        ## TRUE POSITIVE RATE COMPUTATION FOR EACH OUTER LOOP (TEST SET)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier_ROC = nested_score['estimator'][split_iter].best_params_["classifier"]
        feature_selector_ROC = ColumnSelector(nested_score['estimator'][split_iter].best_params_["feature_selector__cols"])
        pipe_ROC = Pipeline(steps=[('scaler', scaler), ('feature_selector', feature_selector_ROC), ('classifier', classifier_ROC)])
        pipe_ROC.fit(X_train, y_train)
        y_pred = pipe_ROC.decision_function(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_auc = auc(fpr, tpr)

        ## SAVING THE FEATURE COMBINATION AUTOMATICALLY SELECTED
        mydict_key = str(nested_score['estimator'][split_iter].best_params_["feature_selector__cols"])
        mydict[mydict_key] = mydict[mydict_key] + 1

        print()
        split_iter = split_iter + 1
##########################################

### PRINTING SOME SCORES TO THE STANDARD OUTPUT
print()
## bal_acc
# TRAIN
bal_acc_train = np.mean(bal_acc_train_scores)
#print ('*** Train: AVERAGE bal_acc', bal_acc_train)
bal_acc_train_std = np.std(bal_acc_train_scores)
#print ('*** Train: STD bal_acc', bal_acc_train_std)
#print()
# TEST
bal_acc = np.mean(bal_acc_test_scores)
#print ('*** Test: AVERAGE bal_acc', bal_acc)
bal_acc_std = np.std(bal_acc_test_scores)
#print ('*** Test: STD bal_acc', bal_acc_std)
#print()
## roc_auc
# TRAIN
roc_auc_train = np.mean(roc_auc_train_scores)
print ('*** Train: AVERAGE roc_auc', roc_auc_train)
roc_auc_train_std = np.std(roc_auc_train_scores)
print ('*** Train: STD roc_auc', roc_auc_train_std)
print()
# TEST
roc_auc = np.mean(roc_auc_test_scores)
print ('*** Test: AVERAGE roc_auc', roc_auc)
roc_auc_std = np.std(roc_auc_test_scores)
print ('*** Test: STD roc_auc', roc_auc_std)
print()
## sensitivity
# TRAIN
sensitivity_train = np.mean(sensitivity_train_scores)
#print ('*** Train: AVERAGE sensitivity', sensitivity_train)
sensitivity_train_std = np.std(sensitivity_train_scores)
#print ('*** Train: STD sensitivity', sensitivity_train_std)
#print()
# TEST
sensitivity = np.mean(sensitivity_test_scores)
#print ('*** Test: AVERAGE sensitivity', sensitivity)
sensitivity_std = np.std(sensitivity_test_scores)
#print ('*** Test: STD sensitivity', sensitivity_std)
#print()
## specificity
# TRAIN
specificity_train = np.mean(specificity_train_scores)
#print ('*** Train: AVERAGE specificity', specificity_train)
specificity_train_std = np.std(specificity_train_scores)
#print ('*** Train: STD specificity', specificity_train_std)
#print()
# TEST
specificity = np.mean(specificity_test_scores)
#print ('*** Test: AVERAGE specificity', specificity)
specificity_std = np.std(specificity_test_scores)
#print ('*** Test: STD specificity', specificity_std)
#print()
##########################################

### SAVING FEATURE COMBINATIONS FREQUENCIES AND PERFORMANCE SCORES IN CSV FILES ###
combs_sel = pd.DataFrame.from_dict(mydict, orient="index", columns = ["#_of_times"])
combs_sel.to_csv("csv_results/CM_CombsSelected.csv")

rows = ["Train_bal_acc_ave",
        "Train_bal_acc_std",
        "Test_bal_acc_ave",
        "Test_bal_acc_std",
        "Train_roc_auc_ave",
        "Train_roc_auc_std",
        "Test_roc_auc_ave",
        "Test_roc_auc_std",
        "Train_sensitivity_ave",
        "Train_sensitivity_std",
        "Test_sensitivity_ave",
        "Test_sensitivity_std",
        "Train_specificity_ave",
        "Train_specificity_std",
        "Test_specificity_ave",
        "Test_specificity_std"]

column = [bal_acc_train,
        bal_acc_train_std,
        bal_acc,
        bal_acc_std,
        roc_auc_train,
        roc_auc_train_std,
        roc_auc,
        roc_auc_std,
        sensitivity_train,
        sensitivity_train_std,
        sensitivity,
        sensitivity_std,
        specificity_train,
        specificity_train_std,
        specificity,
        specificity_std
        ]

Scores = pd.DataFrame(index = rows, columns=["Score"])

Scores['Score'] = column
Scores.to_csv("csv_results/CM_Scores.csv")
##########################################

### SAVING TRUE POSITIVE RATE SCORES IN A CSV FILE (# of rows = NUM_TRIALS * NUM_SPLITS, # of columns = 1000)
tpr_df = pd.DataFrame (tprs, columns=None, index=None)
tpr_df.to_csv("csv_results/CM_tpr.csv")
##########################################

### PLOTTING AND SAVING ROC CURVE ###
plt.plot([0, 1], [0, 1], '--', color='r', label='Random classifier', lw=2, alpha=0.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC', lw=2, alpha=0.8)

## Standard deviation computation
std_tpr = np.std(tprs, axis=0)
tprs_upper_std = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower_std = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower_std, tprs_upper_std, color='green', alpha=.2,label=r'$\pm$ 1 SD')

## 99.9% CI computation
z = 3.291
SE = std_tpr / np.sqrt(NUM_TRIALS * NUM_SPLITS)
tprs_upper_95CI = mean_tpr + (z * SE)
tprs_lower_95CI = mean_tpr - (z * SE)
plt.fill_between(mean_fpr, tprs_lower_95CI, tprs_upper_95CI, color='grey', alpha=.5,label=r'$\pm$ 99.9% CI')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.axis('square')
plt.savefig("Figures/CM_prova_mean_ROC_with_std_99.9CI_Grace_square.png", dpi=600)
#plt.close()
##########################################

# Elapsed time
elapsed_time = time.time() - start_time
# total time taken
print("Runtime of the program is", elapsed_time)
