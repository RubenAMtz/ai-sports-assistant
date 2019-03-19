from sklearn import ensemble
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict
from joblib import load, dump
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.base import clone


def select_from_model(X, y):
    classifier = SGDClassifier()
    sfm = SelectFromModel(classifier, threshold=0.25)
    sfm.fit(X, y)
    n_features = sfm.transform(X).shape[1]

    while n_features > 10:
        sfm.threshold += 10
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]
        # print(n_features)

    return X_transform




class TestClassifiers(object):

    def __init__(self, X, y, classifiers=None, scoring="accuracy"):
        
        self.X = X
        self.y = y
        
        if classifiers is None:
            self.classifiers = []
        self.classifiers = classifiers
        
        self.scoring = scoring
        
        # attribute asigned in call_fit method
        self.scores = []
        self.preds = []

    
    def cv_on_classifiers(self, folds=5):
        """ Apply cross validation on all classifiers
        """

        for classifier in self.classifiers:
            print("Score for Cross-Validation on {} Classifier".format(type(classifier).__name__))
            score = cross_val_score(classifier, self.X, self.y, scoring=self.scoring, cv=folds)
            print("Score:", score, "Mean score:", np.mean(score))
            self.scores.append(np.mean(score))
    
    def cv_pred_on_classifiers(self, folds=5):
        """ Apply cross validation predict on all classifiers
        """

        for classifier in self.classifiers:
            print("Predictions for Cross-Validation on {} Classifier".format(type(classifier).__name__))
            preds = cross_val_predict(classifier, self.X, self.y, cv=folds, n_jobs=5)
            self.preds.append(preds)


    def fit_on_classifiers(self, dump_=True):
        """ Apply fit method to all classifiers
        """

        for classifier in self.classifiers:
            print("Fitting {} Classifier".format(type(classifier).__name__))
            classifier.fit(self.X, self.y)
            print("Finished training classifier ", str(classifier))
            
            if dump_:
                dump(classifier, "models/{}.joblib".format(type(classifier).__name__))
    

    def best_classifier(self):
        """ Returns the classifier with the max score
        """
        max_ = self.scores.index(np.max(self.scores))
        return self.classifiers[max_]


    def load_classifiers(self):
        """ Load classifiers from file
        """
        for classifier in self.classifiers:
            classifier = load("models/{}.joblib".format(type(classifier).__name__))
    
    # @property
    # def classifiers(self):
    #     # clfs = []
    #     # for classifier in self.classifiers:
    #     #     clfs.append(clone(classifier))
    #     # return clfs
    #     return self.classifiers
    
# model = Sequential()
# model.add(Dense(input_dim=48, units=48))
# model.add(Activation("relu"))
# model.add(Dropout(0.2))
# model.add(Dense(units=48))
# model.add(Activation("relu"))
# model.add(Dropout(0.2))
# model.add(Dense(units=30))
# model.add(Activation("relu"))
# model.add(Dropout(0.2))
# model.add(Dense(units=7))
# model.add(Activation("softmax"))

# sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=keras.optimizers.adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# #one hot encoding:
# oh_encoder = OneHotEncoder(categories=[['start','take-off','power-position','extension', 'reception-snatch','reception-clean','jerk']])
# y_train = y_train.values.reshape(-1,1)
# oh_encoder.fit(y_train)
# y_train_wide = oh_encoder.transform(y_train).toarray()
# y_val_wide = oh_encoder.transform(y_val).toarray()

# model.fit(X_train, np.asfarray(y_train_wide), \
#           epochs=20, batch_size=32, verbose=1, \
#           validation_data=(X_val, y_val_wide))

# print("****** Validation Data ********")
# # Make a set of predictions for the validation data
# y_pred = model.predict_classes(X_val)

# #label encoder:
# encoder = LabelEncoder()
# encoder.fit(['start','take-off','power-position','extension','reception-snatch','reception-clean','jerk'])
# y_val = encoder.transform(y_val)
# print(y_val)

# # Print performance details
# print(metrics.classification_report(y_val, y_pred))

# # Print confusion matrix
# print("Confusion Matrix")
# display(pd.crosstab(y_val, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# oh_encoder = OneHotEncoder(categories=[one_hot_encoding_cats])
# y_train = y.values.reshape(-1,1)
# oh_encoder.fit(y_train)
# y_train_wide = oh_encoder.transform(y_train).toarray()

