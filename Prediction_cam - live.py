import cv2
import numpy as np
import tensorflow as tf, sys
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import time
import pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
import time
import pygame
from threading import Thread

##################### Fonctions : graph + extraction ############################

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

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor,
            {'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
            labels.append(label)

        return features, labels
    
    
##################### WEBCAM  ###################################################

        
# Import du SVM 

clf = joblib.load('svm.pkl')

flux = cv2.VideoCapture(2)

font = cv2.FONT_HERSHEY_SIMPLEX

_ ,img = flux.read()

utc = time.time()


while (True):

    images_test =[]

    cv2.rectangle(img,(170,50),(510,450),(0,205,0),2)
    
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)

    _ ,img = flux.read()

    now = time.time()

    #if key == ord('s'): [Mode manuel]
    
    if now-utc > 1.5 :

        _ ,img_predict = flux.read()

        img_predict = img_predict[50:450, 170:510]

        path = 'test/image_test.jpg'

        cv2.imwrite(path,img_predict)

        images_test.append(path)

        print("Image saved")

        #flux.release()

        features,labels = extract_features(images_test,'test')

        print('Processing image...')

        #flux = cv2.VideoCapture(2)
                
        X = features
        y_pred = clf.predict(X)

        cv2.putText(img_predict,y_pred[0],(0,25), font, 1 ,(20, 217, 255), 2, cv2.LINE_AA)
        cv2.imshow('Prediction', img_predict)
       
        utc = time.time()
                


    if key == 0x1b:


        cv2.destroyAllWindows()

        flux.release()
        break

