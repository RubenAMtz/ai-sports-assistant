from joblib import load, dump
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import load_model


# scaler = load('models/models with distances/scaler_not_augmented.joblib')

def pose_inference(query):
    
    scaler = load('./models/scaler_augmented_normalized.joblib')
    query = scaler.transform(query)
    classifier = load('./models/pose/XGBClassifier_augmented_normalized.joblib')
    pred = classifier.predict(query)
    return pred



def grade_inference(query):

    scaler = load('./models/scaler_augmented_normalized.joblib')
    query = scaler.transform(query)
    model = load_model('./models/grade/deep_model.h5')
    pred = model.predict(query)
    return pred