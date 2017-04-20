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

images_dir_huaweimate9 = 'images/huawei_mate_9/'
images_dir_iphone7plus = 'images/iphone_7_plus/'
images_dir_meizum3s = 'images/meizu_m3s/'
images_dir_samsunggalaxys4 = 'images/samsung_galaxy_s4/'
images_dir_human = 'images/human/'
images_dir_iphone5 = 'images/iphone_5/'
images_dir_huaweip8lite = 'images/huawei_p8_lite/'
images_dir_objetnonreconnu = 'images/objet_non_reconnu/'
images_dir_samsunggalaxys7 = 'images/samsung_galaxy_s7/'
images_dir_hpelitebookg3 = 'images/hp_elitebook_g3/'
images_dir_chaise = 'images/chaise/'
images_dir_extincteur = 'images/extincteur/'
images_dir_telfixe = 'images/telephone_fixe/'
images_dir_telecommande = 'images/telecommande/'
images_dir_gobelet = 'images/gobelet/'
images_dir_xiaomiredmi3pro = 'images/xiaomi_redmi_3pro/'



list_images_1 = [images_dir_huaweimate9+f for f in os.listdir(images_dir_huaweimate9) if re.search('jpg|JPG', f)]
list_images_2 = [images_dir_iphone7plus+f for f in os.listdir(images_dir_iphone7plus) if re.search('jpg|JPG', f)]
list_images_3 = [images_dir_meizum3s+f for f in os.listdir(images_dir_meizum3s) if re.search('jpg|JPG', f)]
list_images_4 = [images_dir_samsunggalaxys4 +f for f in os.listdir(images_dir_samsunggalaxys4) if re.search('jpg|JPG', f)]
list_images_5 = [images_dir_human +f for f in os.listdir(images_dir_human) if re.search('jpg|JPG', f)]
list_images_6 = [images_dir_iphone5 +f for f in os.listdir(images_dir_iphone5) if re.search('jpg|JPG', f)]
list_images_8 = [images_dir_huaweip8lite +f for f in os.listdir(images_dir_huaweip8lite) if re.search('jpg|JPG', f)]
list_images_9 = [images_dir_objetnonreconnu +f for f in os.listdir(images_dir_objetnonreconnu) if re.search('jpg|JPG', f)]
list_images_10 = [images_dir_samsunggalaxys7 +f for f in os.listdir(images_dir_samsunggalaxys7) if re.search('jpg|JPG', f)]
list_images_11 = [images_dir_hpelitebookg3 +f for f in os.listdir(images_dir_hpelitebookg3) if re.search('jpg|JPG', f)]
list_images_12 = [images_dir_chaise +f for f in os.listdir(images_dir_chaise) if re.search('jpg|JPG', f)]
list_images_13 = [images_dir_extincteur +f for f in os.listdir(images_dir_extincteur) if re.search('jpg|JPG', f)]
list_images_14 = [images_dir_telfixe +f for f in os.listdir(images_dir_telfixe) if re.search('jpg|JPG', f)]
list_images_15 = [images_dir_telecommande +f for f in os.listdir(images_dir_telecommande) if re.search('jpg|JPG', f)]
list_images_16 = [images_dir_gobelet +f for f in os.listdir(images_dir_gobelet) if re.search('jpg|JPG', f)]
list_images_17 = [images_dir_xiaomiredmi3pro +f for f in os.listdir(images_dir_xiaomiredmi3pro) if re.search('jpg|JPG', f)]




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

i=1

def export_features(feature, label):
    
    global i 
    feature = joblib.dump(feature, 'feature' +str(i) +'.pkl')
    label = joblib.dump(label, 'label' +str(i) +'.pkl')
    i=i+1
    

features_1, labels_1 = extract_features(list_images_1, 'Huawei mate 9')
export_features(features_1, labels_1)

features_2, labels_2 = extract_features(list_images_2, 'Iphone 7 plus')
features_3, labels_3 = extract_features(list_images_3, 'Meizu m3s')
features_4, labels_4 = extract_features(list_images_4, 'Samsung galaxy s4')
features_5, labels_5 = extract_features(list_images_5, 'Bonjour humain !')
features_6, labels_6 = extract_features(list_images_6, 'Iphone 5')
features_8, labels_8 = extract_features(list_images_8, 'Huawei p8 lite')
features_9, labels_9 = extract_features(list_images_9, 'Objet non identifié')
features_10, labels_10 = extract_features(list_images_10, 'Samsung galaxy s7')
features_11, labels_11 = extract_features(list_images_11, 'Hp elitebook g3')
features_12, labels_12 = extract_features(list_images_12, 'Chaise')
features_13, labels_13 = extract_features(list_images_13, 'Extincteur')
features_14, labels_14 = extract_features(list_images_14, 'Telephone fixe')
features_15, labels_15 = extract_features(list_images_15, 'Telecommande')
features_16, labels_16 = extract_features(list_images_16, 'Gobelet')
features_17, labels_17 = extract_features(list_images_17, 'Xiaomi redmi 3 pro')


X_train= np.concatenate((features_1[:30],features_2[:30],features_3[:30],features_4[:30], features_5[:30], features_6[:30], features_8[:30], features_9[:50], features_10[:30], features_11,features_12, features_13, features_14, features_15[:30], features_16[:30], features_17[:30]),axis=0)
y_train= np.concatenate((labels_1[:30],labels_2[:30],labels_3[:30],labels_4[:30], labels_5[:30], labels_6[:30], labels_8[:30], labels_9[:50], labels_10[:30], labels_11, labels_12, labels_13, labels_14, labels_15[:30], labels_16[:30], labels_17[:30]), axis=0)


X_test= np.concatenate((features_1[30:],features_2[30:],features_3[30:],features_4[30:], features_5[30:], features_6[30:], features_8[30:], features_9[50:], features_10[30:], features_11, features_12, features_13, features_14, features_15[30:], features_16[30:], features_17[30:] ),axis=0)
y_test= np.concatenate((labels_1[30:],labels_2[30:],labels_3[30:],labels_4[30:], labels_5[30:], labels_6[30:], labels_8[30:], labels_9[50:], labels_10[30:], labels_11, labels_12, labels_13, labels_14, labels_15[30:], labels_16[30:], labels_17[30:]), axis=0)


#X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=1)

clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


#Export the trained SVM
svm = joblib.dump(clf, 'svm.pkl')


print("Test accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))



