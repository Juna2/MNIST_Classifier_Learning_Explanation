import tensorflow as tf
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
tf.set_random_seed(777)

xy = np.loadtxt('function_data.csv', delimiter=' ', dtype=np.float32)
# x_data = xy[:, 0]
#열벡터를 열행렬로 바꿔준다
x_data = np.array(xy[:, 0], ndmin=2).T
y_data = np.array(xy[:, -1], ndmin=2).T

x_train, x_test, y_train, y_test = \
train_test_split(x_data, y_data, test_size= 0.3, random_state=10)


X = tf.placeholder(tf.float32, shape=[None, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([1, 3]), name='weight1')
b1 = tf.Variable(tf.random_normal([3]), name='bias1')

'''Hidden layer에 있는 activation function은 classification 알고리즘이던 
regression 알고리즘이던 상관없이 sigmoid와 같은 비선형적인 함수가 들어가야한다.'''
layer1 = tf.sigmoid(X * W1 + b1)
# layer1 = X * W1 + b1

W2 = tf.Variable(tf.random_normal([3, 3]), name='weight2')
b2 = tf.Variable(tf.random_normal([3]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
# layer2 = tf.matmul(layer1, W2) + b2

W3 = tf.Variable(tf.random_normal([3, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
'''출력은 연속적인 값이므로 마지막 레이어에는 identify function을 사용한다'''
hypothesis = tf.matmul(layer2, W3) + b3

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(10000):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W1, b1, train],
                    feed_dict = {X: x_train, Y: y_train})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)


y_pred = sess.run(hypothesis, feed_dict={X: x_test})
graph = plt.plot(x_train, y_train, "o", alpha=0.7)
graph = plt.plot(x_test, y_pred, "ro")
plt.show(graph)
