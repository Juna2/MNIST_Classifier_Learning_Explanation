import tensorflow as tf

a = tf.constant([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 8.0, 4.0], [9.0, 2.0, 3.0, 4.0]])
Y = tf.constant([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
# c = tf.nn.softmax_cross_entropy_with_logits(logits=a, labels=Y)
# c = tf.reduce_mean(b)
# b = tf.constant([3.0, 1.0], shape=[1,2])
# c = tf.matmul(a, b)
# d = tf.Print(c, [c], message="\n")

with tf.Session() as sess:
    print(sess.run(a)[1, :])
    # sess.run(d)