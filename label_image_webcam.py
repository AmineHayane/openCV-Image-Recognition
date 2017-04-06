import cv2
import numpy as np
import tensorflow as tf, sys
import time



font = cv2.FONT_HERSHEY_SIMPLEX
 
#Flux web cam
flux = cv2.VideoCapture(0)


#Prise de photo
_ ,img = flux.read()


while (True):

    cv2.rectangle(img,(170,50),(510,450),(0,205,0),2)
    
    cv2.imshow('Image', img)

    
    key = cv2.waitKey(1)
    
    _ ,img = flux.read()

    if key == ord('d'):

        _ ,img_predict2 = flux.read()

        img_predict2 = img_predict2[50:450, 170:510]

        cv2.imwrite('input_image2.jpg', img_predict2)

        flux.release()        

        # Read in the image_data
        image_data = tf.gfile.FastGFile('input_image2.jpg', 'rb').read()

        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line 
                           in tf.gfile.GFile("retrained_labels_smartphones.txt")]

        # Unpersists graph from file
        with tf.gfile.FastGFile("retrained_graph_smartphones.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            prediction_label=[]
            prediction_score=[]
            
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                prediction_label.append(human_string)
                prediction_score.append(score)
                print('%s (score = %.5f)' % (human_string, score))
                
            flux = cv2.VideoCapture(0)

            prediction2=prediction_label[0]
            score2= prediction_score[0]
            
            #Write the prediction on the image
            
##            cv2.putText(img_predict2,str(prediction_label[0]),(0,25), font, 1 , (0,0,0), 2, cv2.LINE_AA)
##
##            cv2.imshow('Prediction2', img_predict2)

        

    if key == ord('s') :

        _ ,img_predict1 = flux.read()

        img_predict1 = img_predict1[50:450, 170:510]

        cv2.imwrite('input_image1.jpg', img_predict1)

        

       

        
##        edged = cv2.Canny(img_predict, 10, 250)
##
##        #applying closing function 
##        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
##        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
##
##        #finding_contours 
##        im2, contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##
##        cv2.drawContours(img_predict, contours, -1, (0,255,0), 2)
##        cv2.imshow('Output', img_predict)
##        
##        cv2.imwrite('input_image.jpg',img_predict)
##
##        idx=0
##        for c in contours:
##            x,y,w,h = cv2.boundingRect(c)
##            if w>50 and h>50:
##                idx+=1
##                new_img=img_predict[y:y+h,x:x+w]
##                cv2.imwrite(str(idx) + '.jpg', new_img)


        flux.release()

       

        # Read in the image_data
        image_data = tf.gfile.FastGFile('input_image1.jpg', 'rb').read()

        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line 
                           in tf.gfile.GFile("retrained_labels_smartphones.txt")]

        # Unpersists graph from file
        with tf.gfile.FastGFile("retrained_graph_smartphones.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            prediction_label=[]
            prediction_score=[]
            
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                prediction_label.append(human_string)
                prediction_score.append(score)
                print('%s (score = %.5f)' % (human_string, score))


            flux = cv2.VideoCapture(0)

            prediction1=prediction_label[0]
            score1= prediction_score [0]
            
            #Write the prediction on the image
            
##            cv2.putText(img_predict1,str(prediction_label[0]),(0,25), font, 1 , (0,0,0), 2, cv2.LINE_AA)
##
##            cv2.imshow('Prediction1', img_predict1)
##

        
        
    if key == 0x0d :
        

         cv2.destroyAllWindows()     


         if (score1 > score2):

             
             final_label = prediction1
             final_score = score1

             if (final_score>0.5):

                 cv2.putText(img_predict1,str(final_label),(0,25), font, 1 , (0,0,0), 2, cv2.LINE_AA)
                 cv2.putText(img_predict1,str(final_score),(0,380), font, 1 , (0,0,0), 2, cv2.LINE_AA)

                 cv2.imshow('Final prediction', img_predict1)

             else:

                 print("La reconnaissance n'a pas pu aboutir, merci de recommencer")

                 
  
            
         else:

             final_label = prediction2
             final_score = score2

             if (final_score > 0.5):

                 cv2.putText(img_predict2,str(final_label),(0,25), font, 1 , (0,0,0), 2, cv2.LINE_AA)
                 cv2.putText(img_predict2,str(final_score),(0,380), font, 1 , (0,0,0), 2, cv2.LINE_AA)

                 cv2.imshow('Final prediction', img_predict2)

             else:

                 print("La reconnaissance n'a pas pu aboutir, merci de recommencer")



    if key == 0x1b:

        cv2.destroyAllWindows()

        flux.release()
        break

         

         
      
        


