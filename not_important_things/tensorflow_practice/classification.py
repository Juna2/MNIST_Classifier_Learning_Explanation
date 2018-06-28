import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
tf.set_random_seed(777)

"""데이터를 불러온다"""
xy = pd.read_csv('iris.data', header=None)
x_data = np.array(xy.iloc[:, 0:-1].values)
y_data = xy.iloc[:, 4].values

for i in range(0, y_data.shape[0]):
    if y_data[i] == 'Iris-setosa':
        y_data[i] = 0
    elif y_data[i] == 'Iris-versicolor':
        y_data[i] = 1
    elif y_data[i] == 'Iris-virginica':
        y_data[i] = 2
    else :
        print('iris name error')

y_data = np.array(y_data, ndmin=2).T

"""데이터를 학습데이터와 실험데이터로 나눈다"""
x_train, x_test, y_train, y_test = \
train_test_split(x_data, y_data, test_size=0.2, random_state=10)

"""placeholder 만들기"""
classes = 3

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.int32, shape=[None, 1])
Y_one_hot = tf.one_hot(Y, classes)  # 행이 데이터개수고 열이 클래스 개수인 행렬이 만들어짐
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, classes]) # -1은 -1이 아닌 행이나 열에 자신을 맞춘다는 뜻
print("reshape", Y_one_hot) 

"""weight과 bias 만들기"""
W = tf.Variable(tf.random_normal([4, classes]), name='weight')
b = tf.Variable(tf.random_normal([classes]), name='bias')

"""activation function 만들기"""
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

'''====================================여기는 학습할때 씀===================================='''
"""cost function 만들기"""
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
'''======================================================================================'''


'''====================================여기는 테스트할때 씀===================================='''
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # cast는 뒤에 소수점을 버려줌
'''======================================================================================='''

iteration = 8000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(iteration):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})        
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    print('\n\n\n==============test==============')
    print('accuracy : ',sess.run(accuracy, feed_dict={X: x_test, Y: y_test}), '%')




