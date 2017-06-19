import os
import re
import cv2
import sys
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
import math
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib

from datetime import datetime, timedelta
import time

path = os.path.dirname(os.path.realpath(__file__))
category = os.path.basename(path)

def training():
    
    list_features = ['features/' +f for f in os.listdir('features/') if re.search('pkl', f)]
    list_labels = ['labels/' +f for f in os.listdir('labels/') if re.search('pkl', f)]

    features = np.empty((1,2048))
    labels =[]

    for feature in list_features:
        
        f = joblib.load(feature)
        features = np.concatenate((features,f),axis=0)

    for label in list_labels:

        l = joblib.load(label)
        labels = np.concatenate((labels,l),axis=0)

    features = np.delete(features, (0), axis=0)

    ############# TRAINING ###################################################

    X_train, y_train =features, labels

    clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)

    
    #Export the trained SVM
    svm = joblib.dump(clf, category + '.pkl')

    print("Training accuracy: {0:0.1f}%".format(accuracy_score(y_train,y_pred)*100))


    #Erreurs de classfication
    misclass = np.where(y_train != y_pred)

    ##for ind in misclass:
    ##        
    ##    print ("Prediction", y_pred[ind], "instead of" , y_train[ind])
    ##  

def doFirst():
    curTime = datetime.now()
    desTime = curTime.replace(hour=00, minute=00, second=00, microsecond=00)
    delta = desTime - curTime
    skipSeconds = (24*60*60+delta.total_seconds())%(24*60*60)
    skipMinutes = (24*60+delta.total_seconds()/60)%(24*60)
    print ("Prochain training dans", skipMinutes, "minutes")
    return skipSeconds

############# Training every day at 00:00  ###################################################

while True:
    training()
    time.sleep(doFirst())   
     
