import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time as t

'''CIFAR 사진 불러옴'''
with open(
    '/home/juna/Documents/my trace/tensorflow/tensorflow_practice/data_batch_1', 'rb') as fo:
    a = pickle.load(fo, encoding='bytes')
with open(
    '/home/juna/Documents/my trace/tensorflow/tensorflow_practice/data_batch_2', 'rb') as fo:
    b = pickle.load(fo, encoding='bytes')
with open(
    '/home/juna/Documents/my trace/tensorflow/tensorflow_practice/data_batch_3', 'rb') as fo:
    c = pickle.load(fo, encoding='bytes')
with open(
    '/home/juna/Documents/my trace/tensorflow/tensorflow_practice/data_batch_4', 'rb') as fo:
    d = pickle.load(fo, encoding='bytes')
with open(
    '/home/juna/Documents/my trace/tensorflow/tensorflow_practice/data_batch_5', 'rb') as fo:
    e = pickle.load(fo, encoding='bytes')
with open(
    '/home/juna/Documents/my trace/tensorflow/tensorflow_practice/test_batch', 'rb') as fo:
    f = pickle.load(fo, encoding='bytes')

'''=================================사진들을 인풋데이터로 만드는 작업================================='''
imgs = np.r_[a[b'data'], b[b'data'], c[b'data'], d[b'data'], e[b'data']]
imgs_test = f[b'data']
'''파일을 가로세로 RGB 형식으로 reshape함'''
imgs = imgs.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8") 
imgs_test = imgs_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8") 
'''reshape(10000, 3, 32, 32)을 행렬이라 생각했을때 인덱스를 각각 0 1 2 3으로 부여하고 transpose()안에 원하는 형태로 인덱스를 나열한 것이다.
그래서 결과적으로 shape은 (10000, 32, 32, 3)이 된다.'''
'''==========================================================================================='''

'''===============================레이블을 원하는 형태로로 만드는 작업==============================='''
labels = np.r_[a[b'labels'], b[b'labels'], c[b'labels'], d[b'labels'], e[b'labels']]
labels_test = f[b'labels']
labels_test = np.array(labels_test)
'''label을 one-hot 형태로 바꾼다'''
labels_one_hot = np.zeros((len(labels), 10))
labels_one_hot_test = np.zeros((len(labels_test), 10))

for i in range(len(labels)):
    for j in range(10):
        if j == labels[i]:
            labels_one_hot[i, j] = 1
        else:
            labels_one_hot[i, j] = 0
'''=========================================================================================='''

'''Visualizing CIFAR 10'''
# fig, axes1 = plt.subplots(5,5,figsize=(3,3))
# for j in range(5):
#     for k in range(5):
#         i = np.random.choice(range(len(imgs))) # 10000개 중에 사진 랜덤으로 뽑기
#         axes1[j][k].set_axis_off() # 축에 눈금이나 수치 표시 안하기
#         axes1[j][k].imshow(imgs[i:i+1][0])
# # plt.show()
'''결과적으로 이미지는 imgs에 레이블은 labels_one_hot에 들어있다.'''

#hyper parameters
learning_rate = 0.001
training_epochs = 40
batch_size = 10

# input place holders
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 32, 32, 3)
'''convnet에서 쓸 weight들을 만들어준다. 여기서 4개의 요소가 있는 벡터는 의미가 
[conv_filter의 가로, conv_filter의 세로, 흑백이면 1이고 칼라면 3, conv_filter의 채널 수(개수)]'''
'''여기서 알아둘 것은 첫번째 convnet의 3번째항은 컬러라 3이지만 이 3개가 합해져서 출력은 1개가 된다는 것이다
따라서 이 convnet의 출력은 [5, 5, 16]이 된다.'''
W1 = tf.Variable(tf.random_normal([5, 5, 3, 16], stddev=0.01))
'''strides에서 1번째와 4번째는 반드시 1이라고 한다. 가운데 두 수가 각각 x,y방향으로 이동하는 칸의 개수'''
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt																				
'''ksize에서 가운데 2수가 각각 pooling하는 영역의 가로, 세로 크기이다 1번째수는 batch고 4번째수는 channel수'''
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

W2 = tf.Variable(tf.random_normal([5, 5, 16, 20], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

W3 = tf.Variable(tf.random_normal([5, 5, 20, 20], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
'''여기서 -1이 의미하는 것은 다른 항을 먼저 구하고 -1인 항은 알아서 구하라는 뜻이다'''
L3_flat = tf.reshape(L3, [-1, 4*4*20])


W4 = tf.get_variable("W4", shape=[4 * 4 * 20, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L3_flat, W4) + b

'''====================================여기는 학습할때 씀===================================='''
'''여기서의 reduce_mean은 사실상 batch를 위한 것이다. 
softmax_cross_entropy_with_logits 함수의 결과는 사진 한개당 숫자하나다'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
'''======================================================================================'''

'''====================================여기는 테스트할때 씀===================================='''
correct_prediction = tf.equal(
    tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # cast는 뒤에 소수점을 버려줌
'''======================================================================================='''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("\n\nLearning started. It's going to take some time.")

    '''epoch은 전체 데이터를 몇바퀴 도는지를 의미'''
    for epoch in range(training_epochs):
        avg_cost = 0
        '''총 레이블의 개수 즉, 데이터의 개수를 batch_size로 나누면 총 batch의 개수 나옴'''
        total_batch = int(len(labels_one_hot)/batch_size)
        # print(total_batch)
        for i in range(total_batch):
            batch_xs = imgs[i*batch_size:(i+1)*batch_size, :, :, :]
            batch_ys = labels_one_hot[i*batch_size:(i+1)*batch_size, :]
            c, _ = sess.run([cost, optimizer], feed_dict={
                            X: batch_xs, Y: batch_ys})
            
            avg_cost += c / total_batch

        print('Epoch : ', '%04d' %(epoch +1),
                'avg_cost =', '{:.9f}'.format(avg_cost),
                'cost =', c)

    print("\n\nLearning finished")
    t.sleep(0.5)
    print('Now testing')

    print('Accuracy : ', sess.run(accuracy, feed_dict={X: imgs_test, Y: labels_one_hot_test}))

