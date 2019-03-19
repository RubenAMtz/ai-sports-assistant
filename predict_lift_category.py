import numpy as np
import pandas as pd
import tqdm
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn import ensemble
from joblib import load, dump
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from IPython.display import display
import os
import glob
from constants import COLUMNS, COLUMNS_NORM
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
import keras
from models import TestClassifiers

# import X data
# this will be a benchmark, will be used to compare against pixel coordinates model
X = pd.read_csv('../data_augmentation/joint_data_3d_distances_norm.csv', index_col = 0)
y = pd.read_csv('../classification/grades_and_categories.csv')
y = y['category']
print(X.shape)
print(y.shape)



augmented_list =  glob.glob('../data_augmentation/*norm.csv')
augmented_X = pd.DataFrame()
augmented_X.columns = range(len(augmented_X.columns))
augmented_y = pd.DataFrame()
for augmented in tqdm.tqdm(augmented_list):
    df_file = pd.read_csv(augmented, index_col = 0)
    df_file.columns = range(len(df_file.columns))
    # df_file.columns = COLUMNS_NORM
    augmented_X = pd.concat([augmented_X, df_file], axis=0, ignore_index=True)
    augmented_y = pd.concat([augmented_y, y], axis=0, ignore_index=True)
if not os.path.isfile('../data_augmentation/augmented_X.csv'):
    augmented_X.to_csv('../data_augmentation/augmented_X.csv', index=False)
if not os.path.isfile('../data_augmentation/augmented_y.csv'):
    augmented_y.to_csv('../data_augmentation/augmented_y.csv', index=False)




print(augmented_X.shape)
print(augmented_y.shape)


X_train, X_val, y_train, y_val = train_test_split(augmented_X, augmented_y, test_size = 0.2, random_state=40)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=40)



scaler = StandardScaler()
scaler.fit(X_train)
dump(scaler, './models/scaler.joblib')
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

# cofigure models:

randforest = ensemble.RandomForestClassifier(n_estimators=1000, criterion="entropy", 
                                                    max_depth=15, max_leaf_nodes=50, n_jobs=5, 
                                                    verbose=1, max_features=0.2, min_samples_split=5,
                                                    min_samples_leaf=5)

sgd = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, 
    fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=5, 
    random_state=None, learning_rate='adaptive', eta0=0.1, power_t=0.5, early_stopping=False, validation_fraction=0.1, 
    n_iter_no_change=10, class_weight=None, warm_start=False, average=False, n_iter=None)

naive = GaussianNB()

ovo_linearSVC = OneVsOneClassifier(LinearSVC(random_state=0), n_jobs=5)

ovr_SGD = OneVsRestClassifier(SGDClassifier(random_state=0), n_jobs=5)

gboosting = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=250, subsample=1.0, 
        criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=4, 
        min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=1, 
        max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

from xgboost import XGBClassifier
xgb = XGBClassifier(predictor='gpu_predictor', learning_rate=0.1, nthread=5, max_depth=6, silent=False, verbosity=3)


clfs = [randforest, sgd, naive, ovo_linearSVC, ovr_SGD, gboosting, xgb]
# clfs_test = [sgd]


"""
# test_clfs = TestClassifiers(X_train, y_train, [ clfs[5] ])
# test_clfs.cv_on_classifiers(folds=5)
# print("Scores:", test_clfs.scores)

# Results from GradientBoostingClassifier w/ 200 estimators, max_depth=3, 5 fold cv

# Score: [0.96560575 0.97225077 0.96993834 0.96606684 0.96992288] Mean score: 0.9687569145136947
# Mean score: 0.9687569145136947
"""
"""
# Results from GradientBoostingClassifier w/ 250 estimators, max_depth=4, 5 fold cv

Score: [0.96791581 0.97327852 0.97250771 0.96580977 0.97377892] Mean score: 0.970658145638928
Scores: [0.970658145638928]

test_clfs = TestClassifiers(X_train, y_train, [ clfs[5] ])
test_clfs.cv_on_classifiers(folds=5)
print("Scores:", test_clfs.scores)

"""

test_clfs = TestClassifiers(X_train, y_train, [ clfs[6] ])
# test_clfs.cv_on_classifiers(folds=5)
# print("Scores:", test_clfs.scores)

test_clfs.fit_on_classifiers()
best_clf = load('./models/XGBClassifier.joblib')
y_predict = best_clf.predict(X_val)
acc = metrics.accuracy_score(y_val, y_predict)
print("Accuracy score (eval): " +  str(acc))
print(metrics.classification_report(y_val, y_predict))


# y_train_predict = cross_val_predict(best_clf, X_train, y_train, cv=5, n_jobs=5)
# classes = ["start", "take-off", "power-position", "extension", "reception-snatch", "reception-clean", "jerk"]
# cmx = confusion_matrix(y_train, y_train_predict, labels=classes)
# cm_df = pd.DataFrame(cmx, columns=classes, index=classes)
# print(cm_df)


# test_clfs.fit_on_classifiers()
# best_clf = load('models/GradientBoostingClassifier.joblib')

# print("Scores:", test_clfs.scores)

# y_predict = best_clf.predict(X_val)
# acc = metrics.accuracy_score(y_val, y_predict)
# print("Accuracy score (eval): " +  str(acc))
# print(metrics.classification_report(y_val, y_predict))




# if not os.path.isfile('gb_model.joblib'):
#     classifier = call_gbclassifier(X = X_train, y = y_train, save_path = 'gb_model.joblib')
# else:
#     classifier = load('gb_model.joblib')

# y_predict = classifier.predict(X_val)
# mae_eval = metrics.accuracy_score(y_val, y_predict)
# print('\n')
# print("Accuracy score (eval): " +  str(mae_eval))
# print('\n')
# # Print performance details
# print(metrics.classification_report(y_val, y_predict))


# X_transformed = select_from_model(augmented_X, augmented_y)

# # print(X_transformed)

# if not os.path.isfile('gb_model_new.joblib'):
#     classifier = call_gbclassifier(X = X_transformed, y = augmented_y, save_path = 'gb_model_new.joblib')
# else:
#     classifier = load('gb_model_new.joblib')

# if not os.path.isfile('ovo_model_new.joblib'):
#     classifier = call_onevsoneclassifier(X = X_train, y = y_train, save_path = 'ovo_model_new.joblib')
# else:
#     classifier = load('ovo_model_new.joblib')

# y_predict = classifier.predict(X_val)
# acc = metrics.accuracy_score(y_val, y_predict)
# print('\n')
# print("Accuracy score (eval): " +  str(acc))
# print('\n')
# # Print performance details
# print(metrics.classification_report(y_val, y_predict))