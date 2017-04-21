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

flux = cv2.VideoCapture(2)

# Import du SVM 

clf = joblib.load('svm.pkl')

list_prediction=[]

font = cv2.FONT_HERSHEY_SIMPLEX
 
#Flux web cam
#flux = cv2.VideoCapture('rtsp://192.168.42.1:554/live')


#Prise de photo
_ ,img = flux.read()


indice = 10
images_test =[]

utc = time.time()


while (True):

    cv2.rectangle(img,(170,50),(510,450),(0,205,0),2)
    
    cv2.imshow('Image', img)

    
    key = cv2.waitKey(1)

    _ ,img = flux.read()

    now = time.time()

    #if key == ord('s'): [Mode manuel]
    
    if indice < 10 and now-utc > 0.3 :

        _ ,img_predict = flux.read()

        img_predict = img_predict[50:450, 170:510]

        path = 'test/'+ 'image' + "_" + str(indice) + ".jpg"

        cv2.imwrite(path,img_predict)

        print("Saving image n°" + str(indice+1) + '...')

        #cv2.imwrite('inputimage.jpg', img_predict)

        images_test.append([path])

        utc = time.time()

        if indice == 9:

            print("Finished !")
        
        indice += 1

        

    if key == ord('s'):

        indice=0
        utc = time.time()
        


    if key == 0x0d:

        flux.release()

        j=0
    
        for image in images_test:

            features,labels = extract_features(image,'test')

            print('Processing image n°' + str(j+1) + '...')
                
            X = features
            y_pred = clf.predict(X)

            list_prediction.append(y_pred[0])

            j=j+1

            #cv2.putText(img_predict,str(y_pred[0]),(0,25), font, 1 , (0,0,0), 2, cv2.LINE_AA)
                
            #cv2.imshow('Prediction'+str(j), img_predict)
                

        flux = cv2.VideoCapture(2)

        images_test=[]

        print(list_prediction)

        ring1=0
        ring2=0
        ring3=0
        ring4=0
        ring5=0
        ring6=0
        ring7=0
        ring8=0
        ring9=0
        ring10=0
        ring11=0
        ring12=0
        ring13=0
        ring14=0
        ring15=0
        ring16=0


        for prediction in list_prediction:

            if prediction =='Huawei mate 9':

                ring1=ring1+1
                
            if prediction == 'Iphone 7 plus':

                ring2=ring2+1

            if prediction == 'Meizu m3s':

                ring3=ring3+1
                 
            if prediction == 'Samsung galaxy s4':

                ring4=ring4+1

            if prediction == 'Bonjour humain !':

                ring5=ring5+1

            if prediction == 'Iphone 5':

                ring6=ring6+1

            if prediction == 'Samsung galaxy s6 edge':

                ring7=ring7+1

            if prediction == 'Huawei p8 lite':

                ring8=ring8+1
                
            if prediction == 'Objet non identifié':

                ring9=ring9+1
                
            if prediction == 'Samsung galaxy s7':

                ring10=ring10+1
                
            if prediction == 'Hp elitebook g3':

                ring11=ring11+1

            if prediction == 'Chaise':

                ring12=ring12+1

            if prediction == 'Extincteur':

                ring13=ring13+1
                
            if prediction == 'Telephone fixe':

                ring14=ring14+1
                
            if prediction == 'Telecommande':

                ring15=ring15+1
                
            if prediction == 'Gobelet':

                ring16=ring16+1
                
        

        def final_prediction(ring,produit):

             if ring == max(rings):

                 best_prediction = produit


                 if sum(rings) !=  0:

                     confidence = int(ring/(sum(rings))*100)

                     print(confidence)

                     if confidence > 70 :

                         print('Final prediction :' + best_prediction )

                         cv2.putText(img_predict,best_prediction,(0,25), font, 1 ,(20, 217, 255), 2, cv2.LINE_AA)

                         cv2.putText(img_predict,str(confidence) + '%' ,(0,380), font, 1 , (20,217,255), 2, cv2.LINE_AA)
                 
                         cv2.imshow('Prediction', img_predict)
                         
                     else:

                         print('La reconnaissance ne peut aboutir, merci de recommencer !')


        rings= [ring1, ring2, ring3, ring4, ring5, ring6, ring7, ring8, ring9, ring10, ring11, ring12, ring13, ring14, ring15, ring16]

        print(rings)

        final_prediction(ring1,'Huawei mate 9')
        final_prediction(ring2,'Iphone 7 plus' )
        final_prediction(ring3,'Meizu m3s')
        final_prediction(ring4,'Samsung galaxy s4')
        final_prediction(ring5,'Bonjour humain !')
        final_prediction(ring6,'Iphone 5')
        final_prediction(ring7,'Samsung galaxy s6 edge')
        final_prediction(ring8,'Huawei p8 lite')
        final_prediction(ring9,'Objet non reconnu')
        final_prediction(ring10,'Samsung galaxy s7')
        final_prediction(ring11,'Hp elitebook g3')
        final_prediction(ring12,'Chaise')
        final_prediction(ring13,'Extincteur')
        final_prediction(ring14,'Telephone fixe')
        final_prediction(ring15,'Telecommande')
        final_prediction(ring16,'Gobelet')

        list_prediction=[]


    
    if key == 0x1b:


        cv2.destroyAllWindows()

        flux.release()
        break
