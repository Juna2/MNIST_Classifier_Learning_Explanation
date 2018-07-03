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

# column = 1
# row = 2
# channel = 3
# act_node_conv2_all = np.array([[]])
# print(act_node_conv2_all.shape)
# print(np.array([[column, row, channel]]).shape)

# act_node_conv2_all = np.c_[act_node_conv2_all, [[column, row, channel]]]

# print(act_node_conv2_all)

# a = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]])
# a = np.resize(a, (3,3))
# a_img = np.stack([a, a, a], axis=2)
# a_img[1,1,0] = 0.2
# a_img[1,1,1] = 0
# a_img[1,1,2] = 0
# plt.imshow(a_img)
# plt.show()
# print(a-1)

# a = np.array([[0.112345678, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]], dtype="i")
# a = np.c_[a, [[1, 2, 3, 4, 5.0]]]
# a = np.resize(a, (2,7))
# print(a)

# a = np.array([1, 2, 3, 4, 5, 6])
# b = a / 0
# print(b)

# def a():
#     b = np.array([1, 2, 3])
#     c = np.array([4, 5, 6])
#     return b, c

# p, q = a()

# print(p, q)

a = 2

print((a==2)+1)