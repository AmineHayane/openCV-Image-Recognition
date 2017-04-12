import cv2
import numpy as np
import tensorflow as tf, sys
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import time
import pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib

######## Fonctions graph + extraction ################

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
    
##################### WEBCAM  ######################################

clf = joblib.load('svm.pkl')

list_prediction=[]

font = cv2.FONT_HERSHEY_SIMPLEX
 
#Flux web cam
#flux = cv2.VideoCapture('rtsp://192.168.42.1:554/live')

flux= cv2.VideoCapture(2)


#Prise de photo
_ ,img = flux.read()

j=0

while (True):

    cv2.rectangle(img,(170,50),(510,450),(0,205,0),2)
    
    cv2.imshow('Image', img)

    
    key = cv2.waitKey(1)

    _ ,img = flux.read()


    if key == ord('s') :


        _ ,img_predict = flux.read()

        img_predict = img_predict[50:450, 170:510]

        cv2.imwrite('inputimage.jpg', img_predict)

        flux.release()

        image=['inputimage.jpg']
            
        features,labels = extract_features(image,'test')
        
        X = features
        y_pred = clf.predict(X)

        list_prediction.append(y_pred[0])

        #cv2.putText(img_predict,str(y_pred[0]),(0,25), font, 1 , (0,0,0), 2, cv2.LINE_AA)
        
        #cv2.imshow('Prediction'+str(j), img_predict)

        flux= cv2.VideoCapture(2)

    if key == 0x0d:

        print(list_prediction)

        ring1=0
        ring2=0
        ring3=0
        ring4=0
        ring5=0

        for prediction in list_prediction:

            if prediction =='huawei mate 9':

                ring1=ring1+1
                
            if prediction == 'iphone 7 plus':

                ring2=ring2+1

            if prediction == 'meizu m3s':

                ring3=ring3+1
                 
            if prediction == 'samsung galaxy s4':

                ring4=ring4+1

            if prediction == 'Bonjour humain !':

                ring5=ring5+1

        last_prediction =''
        
        def final_prediction(ring,produit):

             if ring == max(rings):

                 last_prediction = produit

                 print('Final prediction :' + last_prediction )

                 if sum(rings) !=  0:

                     confidence = ring/(sum(rings))*100

                     print(confidence)

                     if confidence > 50 :

                         cv2.putText(img_predict,last_prediction,(0,25), font, 1 , (0,0,0), 2, cv2.LINE_AA)

                         #cv2.putText(img_predict,str(confidence) + '%' ,(0,380), font, 1 , (0,0,0), 2, cv2.LINE_AA)
                 
                         cv2.imshow('Prediction', img_predict)
                         
                     else:

                         print('La reconnaissance ne peut aboutir, merci de recommencer !')


        rings= [ring1, ring2, ring3, ring4, ring5]

        print(rings)

        final_prediction(ring1,'huawei mate 9')
        final_prediction(ring2,'iphone 7 plus' )
        final_prediction(ring3,'meizu m3s')
        final_prediction(ring4,'samsung galaxy s4')
        final_prediction(ring5,'Bonjour humain !')

        list_prediction=[]


    
    if key == 0x1b:


        cv2.destroyAllWindows()

        flux.release()
        break
