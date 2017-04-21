import os
import re
import cv2
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib 

#Images d'entrainement

images_dir_newclass = 'images/ecouteurs/'

list_images = [images_dir_newclass +f for f in os.listdir(images_dir_newclass) if re.search('jpg|JPG', f)]


def create_graph():
    with gfile.FastGFile('retrained_graph_smartphones.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


create_graph()


def extract_features(list_images, label):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):

            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            print('Extracting features from ' + image) 

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor,
            {'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
            labels.append(label)

        return features, labels



def export_features(feature, label, i):
    
    feature = joblib.dump(feature, './features/feature' +str(i) +'.pkl')
    label = joblib.dump(label, './labels/label' +str(i) +'.pkl')

def load_features(i):

        features = joblib.load('features/feature'+ str(i) + '.pkl')
        labels = joblib.load('labels/label' +str(i) + '.pkl')

        return features, labels

        
#Load features & labels
features_1, labels_1 = load_features(1)
features_2, labels_2 = load_features(2)
features_3, labels_3 = load_features(3)
features_4, labels_4 = load_features(4)
features_5, labels_5 = load_features(5)
features_6, labels_6 = load_features(6)
features_8, labels_8 = load_features(7)
features_9, labels_9 = load_features(8)
features_10, labels_10 = load_features(9)
features_11, labels_11 = load_features(10)
features_12, labels_12 = load_features(11)
features_13, labels_13 = load_features(12)
features_14, labels_14 = load_features(13)
features_15, labels_15 = load_features(14)
features_16, labels_16 = load_features(15)
features_17, labels_17 = load_features(16)



#Features for the new class 
features_18, labels_18 = extract_features(list_images, 'Ecouteurs')
export_features(features_18, labels_18, 17)


X_train= np.concatenate((features_1[:30],features_2[:30],features_3[:30],features_4[:30], features_5[:30], features_6[:30], features_8[:30], features_9[:50], features_10[:30], features_11,features_12, features_13, features_14, features_15[:30], features_16[:30], features_17[:30], features_18[:30]),axis=0)
y_train= np.concatenate((labels_1[:30],labels_2[:30],labels_3[:30],labels_4[:30], labels_5[:30], labels_6[:30], labels_8[:30], labels_9[:50], labels_10[:30], labels_11, labels_12, labels_13, labels_14, labels_15[:30], labels_16[:30], labels_17[:30], labels_18[:30]), axis=0)


X_test= np.concatenate((features_1[30:],features_2[30:],features_3[30:],features_4[30:], features_5[30:], features_6[30:], features_8[30:], features_9[50:], features_10[30:], features_11, features_12, features_13, features_14, features_15[30:], features_16[30:], features_17[30:], features_18[30:] ),axis=0)
y_test= np.concatenate((labels_1[30:],labels_2[30:],labels_3[30:],labels_4[30:], labels_5[30:], labels_6[30:], labels_8[30:], labels_9[50:], labels_10[30:], labels_11, labels_12, labels_13, labels_14, labels_15[30:], labels_16[30:], labels_17[30:], labels_18[30:]), axis=0)


clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


#Export the trained SVM
svm = joblib.dump(clf, 'svm-test.pkl')


print("Test accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))



