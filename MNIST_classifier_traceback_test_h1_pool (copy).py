import MNIST_classifier as third
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import struct
import gzip
import sys
import os


random_seed = 123
np.random.seed(random_seed)



def print_(name):
    print(name, ' : \n', eval(name))




third.unzip('ubyte.gz')

X_data = np.array([[]])
y_data = np.array([[]])
X_data, y_data = third.LoadTrainData('./', 'train') # or 'train'
X_test_cand, y_test_cand = third.LoadTrainData('./', 't10k') # or 'train'

X_train, y_train = X_data[:55000,:], y_data[:55000]
X_valid, y_valid = X_data[55000:,:], y_data[55000:]

X_test_cand = X_test_cand[np.argsort(y_test_cand)]
y_test_cand = y_test_cand[np.argsort(y_test_cand)]

tmp = 0
StartIndex = np.array([0])
for index, i in enumerate(y_test_cand):
    if i != tmp:
        StartIndex = np.c_[StartIndex, np.array([index])]
        tmp += 1
      

print(StartIndex)
TestNumber = 7 ####
DifferentOne = 0 ####
PickOne = StartIndex[0,TestNumber]+DifferentOne
X_test, y_test = X_test_cand[PickOne:PickOne+1,:], y_test_cand[PickOne:PickOne+1]



mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_valid_centered = X_valid - mean_vals
X_test_centered = (X_test - mean_vals)/std_val

del X_data, y_data, X_train, X_valid, X_test



g2 = tf.Graph()
with g2.as_default():
    tf.set_random_seed(random_seed)
    
    third.build_cnn()

    saver = tf.train.Saver()


with tf.Session(graph=g2) as sess:

    third.load(saver, sess, epoch=2, path='./model/')

    feed = {'tf_x:0': X_test_centered, 
            'fc_keep_prob:0': 1.0}

    image, Conv1Kernel, Conv1Bias, conv_1_result, result_of_h1_pool, \
    Conv2Kernel, Conv2Bias, conv_2_result, fc_3_w, h2_pool, fc_3_b, fc_4_w, \
    fc_3_result, fc_4_b, fc_4_result, preds = sess.run(['tf_x:0', 'conv_1/_weights:0', \
    'conv_1/_biases:0', 'conv_1/activation:0', 'h1_pool:0', 'conv_2/_weights:0', 'conv_2/_biases:0',\
        'conv_2/activation:0', 'fc_3/_weights:0', 'h2_pool:0', 'fc_3/_biases:0', 'fc_4/_weights:0',\
    'fc_3/activation:0', 'fc_4/_biases:0', 'fc_4/net_pre-activation:0','labels:0'], feed_dict=feed)

    image = np.resize(image, (1,28,28))

                



PixelsToBeRedFc3, Fc3RedValue = third.activated_nodes(fc_3_result, np.array([fc_4_w[:,TestNumber]]), 1) #@@@@




PixelsToBeRedH2pool = third.Fcl2PoolingLayer(PixelsToBeRedFc3, fc_3_w, h2_pool, Fc3RedValue)

PixelsToBeRedH2pool = third.CollectRedValues(PixelsToBeRedH2pool, h2_pool)




PixelsToBeRedConv2 = third.unpooling(PixelsToBeRedH2pool, conv_2_result)

# third.ShowImage(conv_2_result, PixelsToBeRedConv2,'Final_image_No1_0.20 .png', threshold=0.2, RowNumber=8, ColNumber=8)




PixelsToBeRedH1pool = third.Deconvolution(PixelsToBeRedConv2, Conv2Kernel, result_of_h1_pool)

PixelsToBeRedH1pool = third.CollectRedValues(PixelsToBeRedH1pool, result_of_h1_pool)

# third.ShowImage(result_of_h1_pool, PixelsToBeRedH1pool,'Final_image_No1_0.20 .png', threshold=0.2, RowNumber=4, ColNumber=8)




PixelsToBeRedConv1 = third.unpooling(PixelsToBeRedH1pool, conv_1_result)

# third.ShowImage(conv_1_result, PixelsToBeRedConv1,'Final_image_No1_0.20 .png', threshold=0.2, RowNumber=4, ColNumber=8)




PixelsToBeRedImage = third.Deconvolution(PixelsToBeRedConv1, Conv1Kernel, image)

PixelsToBeRedImage = third.CollectRedValues(PixelsToBeRedImage, image)




third.ShowImage(image, PixelsToBeRedImage,'Final_image_No1_0.20 .png', threshold=0.45)