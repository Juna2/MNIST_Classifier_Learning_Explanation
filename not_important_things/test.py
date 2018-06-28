#!/usr/bin/env python
# def print_(name):
#     print(name, ' : ', eval(name))


# a = 344960112 + 31360112 + 62720112
# print_("a")

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np



# img = np.zeros((28, 28),dtype=float)


# plt.figure()
# plt.imshow(img)
# plt.show()

# a = np.array([[1, 2, 3, 4, 5]])
# # b = np.array([[1], [2], [3]])
# # c = a.T*b
# a = a[0,:]
# sorted_index = np.argsort(a)[::-1]
# np.c_[check_point, index] 

# # for i in list(a[0,:]):
# #     print(i)
# # print(np.array([a]).T)



# a =[1, 2, 3, 4, 5]
# sorted_index = np.argsort(a)[::-1]

# sum_1 = 0
# for index, i in enumerate(list(a[sorted_index])):
#     sum_1 += i
#     if sum_1 > 6:
# np.set_printoptions(threshold=np.inf)
# a = np.arange(1024)
# a = np.resize(a, (1,4,4,64))
# print(a)

# for i in range(64):
#     print(a[0,:,:,i])

# np.set_printoptions(threshold=1000)

# g2 = tf.Graph()
# with g2.as_default():
#     a = np.arange(1024)
#     a = np.resize(a, (1,4,4,64))
#     a = tf.constant(a)
#     input_shape = a.get_shape().as_list()[1:]
#     n_input_units = np.prod(input_shape)
#     a_reshaped = tf.reshape(a,shape=(-1, n_input_units))
#     print(a.get_shape())
# #     n_input_units = np.prod(input_shape)

# #     a_reshaped = tf.reshape(a, shape=(-1, n_input_units))

# z = input('pass 1 ?')
# with tf.Session(graph=g2) as sess:
#     z = input('pass 2 ?')
#     a_rs = \
#     sess.run(a_reshaped)
#     np.set_printoptions(threshold=np.inf)
#     print(a_rs)
#     np.set_printoptions(threshold=1000)

for i in range(3,8):
    print(i)