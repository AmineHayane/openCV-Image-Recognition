import numpy as np
import tensorflow as tf, sys
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
import cv2


clf = joblib.load('phones_prctl2.pkl')

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

#################### PREDICTION ################################################

#Image path
images_test =[]
path = 'image_test.jpg'
img_predict = cv2.imread(path)

L = img_predict.shape[0]
l = img_predict.shape[1]

print(L/l)

# (340,453) -> Ratio 4:3
# (340, 604) -> Ratio 16:9

#Ratio 16:9
if (1.7 < (L/l) < 1.8):

    img_predict = cv2.resize(img_predict,(X,X), interpolation = cv2.INTER_CUBIC)

#Ratio 4:3
if (1.3 < (L/l) < 1.4):

     img_predict = cv2.resize(img_predict,(340,453), interpolation = cv2.INTER_CUBIC)

    
cv2.imwrite('image_reduced.jpg',img_predict)

images_test.append('image_reduced.jpg')

#Features extraction
features,labels = extract_features(images_test,'')

#Prediction
X = features
prediction = clf.predict(X)

print(str(prediction[0]))


