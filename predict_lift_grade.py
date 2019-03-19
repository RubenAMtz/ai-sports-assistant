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

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
import keras
from models import TestClassifiers

# import X data
# this will be a benchmark, will be used to compare against pixel coordinates model
X = pd.read_csv('../data_augmentation/joint_data_3d_distances_norm.csv', index_col = 0)
y = pd.read_csv('../classification/grades_and_categories.csv')
y = y['grade']
print(X.shape)
print(y.shape)



augmented_list =  glob.glob('../data_augmentation/*norm.csv')
augmented_X = pd.DataFrame()
augmented_X.columns = range(len(augmented_X.columns))
augmented_y = pd.DataFrame()
for augmented in tqdm.tqdm(augmented_list):
    # df_file = pd.read_csv(augmented, index_col = 0)
    # df_file.columns = range(len(df_file.columns))
    # df_file.columns = COLUMNS_NORM
    # augmented_X = pd.concat([augmented_X, df_file], axis=0, ignore_index=True)
    augmented_y = pd.concat([augmented_y, y], axis=0, ignore_index=True)
# if not os.path.isfile('../data_augmentation/augmented_X.csv'):
#     augmented_X.to_csv('../data_augmentation/augmented_X.csv', index=False)
if not os.path.isfile('../data_augmentation/augmented_y_grade.csv'):
    augmented_y.to_csv('../data_augmentation/augmented_y_grade.csv', index=False)

augmented_X = pd.read_csv('../data_augmentation/augmented_X.csv')


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
xgb = XGBClassifier(predictor='gpu_predictor', learning_rate=0.01, nthread=5, max_depth=8, silent=False, 
                    verbosity=3, booster='gblinear', reg_alpha=0.1, feature_selector='shuffle', eval_metric='mae',
                    updater='coord_descent')


clfs = [randforest, sgd, naive, ovo_linearSVC, ovr_SGD, gboosting, xgb]
# clfs_test = [sgd]

"""
test_clfs = TestClassifiers(X_train, y_train, [ clfs[6] ], scoring='neg_mean_squared_error')
test_clfs.cv_on_classifiers(folds=5)
print("Scores:", test_clfs.scores)

Score: [-0.90174041 -0.90854207 -0.90435509 -0.9021006  -0.90486285] Mean score: -0.9043202037264585
Scores: [-0.9043202037264585]
"""
# test_clfs = TestClassifiers(X_train, y_train, [ clfs[6] ], scoring='neg_mean_squared_error')
# test_clfs.cv_pred_on_classifiers(folds=5)
# y_pred_cv =test_clfs.preds[0]
# plt.plot(range(len(X_train)), y_pred_cv)
# plt.plot(range(len(X_train)), y_train)
# plt.show()

# test_clfs = TestClassifiers(X_train, y_train, [ clfs[6] ], scoring='neg_mean_squared_error')
# test_clfs.fit_on_classifiers()
# best_clf = load('./models/XGBClassifier.joblib')
# y_predict = best_clf.predict(X_val)
# mse = metrics.mean_squared_error(y_val, y_predict)
# print("Mean squared error (eval): " +  str(mse))
if not os.path.isfile('./models/deep_model.h5'):
        

    model = Sequential()
    model.add(Dense(input_dim=276, units=200))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=48))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=30))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation("linear"))

    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=keras.optimizers.adam(), loss='mse', metrics=['mae'])
    # model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
    history = model.fit(X_train, np.asfarray(y_train), epochs=20, batch_size=32, verbose=1, validation_data=(X_val, y_val))

    model.save('./models/deep_model.h5')
    keras.utils.print_summary(model)
else:
    model = load_model('./models/deep_model.h5')
    keras.utils.print_summary(model)

print("****** Validation Data ********")
score = model.evaluate(X_val, y_val, verbose=0)
print('Val loss:', score)
print(model.metrics_names)

y_predict = model.predict(X_val, verbose=1)

title = "Y_predict vs Y_Validation" 
fig, ax1 = plt.subplots(figsize=(12, 8))
plt.title(title)
plt.plot(y_val.values, color='r')
ax1.set_ylabel('y_validation', color='r')
plt.legend(['prediction'], loc=(0.01, 0.95))
plt.plot(y_predict, color='b')
ax2 = ax1.twinx()
ax2.set_ylabel('y_prediction', color='b')
plt.legend(['groundT'], loc=(0.01, 0.9))
plt.grid(True)
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

