import tensorflow as tf
import numpy as np
import struct
import gzip
import sys
import os


learning_rate = 1e-4
random_seed = 123
np.random.seed(random_seed)

def print_(name):
    print(name, ' : ', eval(name))



def conv_layer(input_tensor, name,
               kernel_size, n_output_channels, 
               padding_mode='SAME', strides=(1, 1, 1, 1)):
    with tf.variable_scope(name):
        ## get n_input_channels:
        ##   input tensor shape: 
        ##   [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1] 

        weights_shape = (list(kernel_size) + 
                         [n_input_channels, n_output_channels])

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_channels]))
        print(biases)
        conv = tf.nn.conv2d(input=input_tensor, 
                            filter=weights,
                            strides=strides, 
                            padding=padding_mode)
        print(conv)
        conv = tf.nn.bias_add(conv, biases, 
                              name='net_pre-activation')
        print(conv)
        conv = tf.nn.relu(conv, name='activation')
        print(conv)
        
        return conv



def fc_layer(input_tensor, name, 
             n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, 
                                      shape=(-1, n_input_units))

        weights_shape = [n_input_units, n_output_units]

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_units]))
        print(biases)
        layer = tf.matmul(input_tensor, weights)
        print(layer)
        layer = tf.nn.bias_add(layer, biases,
                              name='net_pre-activation')
        print(layer)
        if activation_fn is None:
            return layer
        
        layer = activation_fn(layer, name='activation')
        print(layer)
        return layer, weights



def build_cnn(norm, lamb):
    ## Placeholders for X and y:
    tf_x = tf.placeholder(tf.float32, shape=[None, 784],
                          name='tf_x')
    tf_y = tf.placeholder(tf.int32, shape=[None],
                          name='tf_y')

    # reshape x to a 4D tensor: 
    # [batchsize, width, height, 1]
    tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1],
                            name='tf_x_reshaped')
    ## One-hot encoding:
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10,
                             dtype=tf.float32,
                             name='tf_y_onehot')

    ## 1st layer: Conv_1
    print('\nBuilding 1st layer: ')
    h1 = conv_layer(tf_x_image, name='conv_1',
                    kernel_size=(5, 5), 
                    padding_mode='VALID',
                    n_output_channels=32)
    ## MaxPooling
    h1_pool = tf.nn.max_pool(h1, 
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], 
                             padding='SAME')
    ## 2n layer: Conv_2
    print('\nBuilding 2nd layer: ')
    h2 = conv_layer(h1_pool, name='conv_2', 
                    kernel_size=(5,5), 
                    padding_mode='VALID',
                    n_output_channels=64)
    ## MaxPooling 
    h2_pool = tf.nn.max_pool(h2, 
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], 
                             padding='SAME')

    ## 3rd layer: Fully Connected
    print('\nBuilding 3rd layer:')
    h3, h3_weights = fc_layer(h2_pool, name='fc_3',
                  n_output_units=1024, 
                  activation_fn=tf.nn.relu)

    ## Dropout
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, 
                            name='dropout_layer')

    ## 4th layer: Fully Connected (linear activation)
    print('\nBuilding 4th layer:')
    h4 = fc_layer(h3_drop, name='fc_4',
                  n_output_units=10, 
                  activation_fn=None)

    ## Prediction
    predictions = {
        'probabilities' : tf.nn.softmax(h4, name='probabilities'),
        'labels' : tf.cast(tf.argmax(h4, axis=1), tf.int32,
                           name='labels')
    }
    
    ## Visualize the graph with TensorBoard:

    ## Loss Function and Optimization
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=h4, labels=tf_y_onehot),
        name='cross_entropy_loss')


    if norm == 'L1':
        L1_norm = tf.norm(h3_weights, ord='fro', axis=(0,1))

        ## Optimizer:
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss+lamb*L1_norm,
                                    name='train_op')

    elif norm == 'L2':
        L2_norm = tf.norm(h3_weights, ord='euclidean', axis=(0,1))

        ## Optimizer:
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss+lamb*L2_norm,
                                    name='train_op')



    ## Computing the prediction accuracy
    correct_predictions = tf.equal(
        predictions['labels'], 
        tf_y, name='correct_preds')

    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32),
        name='accuracy')    

def save(saver, sess, epoch, path='./model/'):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving model in %s' % path)
    saver.save(sess, os.path.join(path,'cnn-model.ckpt'),
               global_step=epoch)

    
def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(
            path, 'cnn-model.ckpt-%d' % epoch))        


keep_prob_param = [0.1, 0.5, 1.0]
norm = ['L1', 'L2']
lamb = [0, 0.1, 0.001]
mini_batch = [1, 4, 16, 64, 128, 1024, 55000]


if (sys.version_info > (3, 0)):
    writemode = 'wb'
else:
    writemode = 'w'

zipped_mnist = [f for f in os.listdir('./')
                if f.endswith('ubyte.gz')]
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read())


path = './'
kind = 'train'
labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

with open(labels_path, 'rb') as lbpath:
    magic, n = struct.unpack('>II', lbpath.read(8))
    y_data = np.fromfile(lbpath, dtype=np.uint8)

with open(images_path, 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
    X_data = np.fromfile(imgpath, dtype=np.uint8).reshape(len(y_data), 784)

print('Rows: %d,  Columns: %d' % (X_data.shape[0], X_data.shape[1]))


kind = 't10k'
labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

with open(labels_path, 'rb') as lbpath:
    magic, n = struct.unpack('>II', lbpath.read(8))
    y_test = np.fromfile(lbpath, dtype=np.uint8)

with open(images_path, 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
    X_test = np.fromfile(imgpath, dtype=np.uint8).reshape(len(y_test), 784)

print('Rows: %d,  Columns: %d' % (X_test.shape[0], X_test.shape[1]))


X_train, y_train = X_data[:55000,:], y_data[:55000]
X_valid, y_valid = X_data[55000:,:], y_data[55000:]

print('Training:   ', X_train.shape, y_train.shape)
print('Validation: ', X_valid.shape, y_valid.shape)
print('Test Set:   ', X_test.shape, y_test.shape)




mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_valid_centered = X_valid - mean_vals
X_test_centered = (X_test - mean_vals)/std_val

print_('sys.getsizeof(X_data)')
print_('sys.getsizeof(y_data)')
print_('sys.getsizeof(X_train)')
print_('sys.getsizeof(X_valid)')
print_('sys.getsizeof(X_test)')
print_('sys.getsizeof(X_train_centered)')
print_('sys.getsizeof(X_valid_centered)')
print_('sys.getsizeof(X_test_centered)')

del X_data, y_data, X_train, X_valid, X_test



# for i in keep_prob_param:
#     for j in norm:
#         for k in lamb:
#             for l in mini_batch:
dropout = keep_prob_param[0]
j = norm[0]
k = lamb[0]
batch_size = mini_batch[0]

random_seed = 123
initialize = True
epochs = 2
shuffle = 'shuffle'

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    
    build_cnn(j, k)
    
    saver = tf.train.Saver()
with tf.Session(graph=g) as sess:
    # train(sess, training_set=(X_train_centered, y_train), batch_size=l,
        # validation_set=(X_valid_centered, y_valid), initialize=True, random_seed=123, dropout=i, epochs=2)



    # X_data = np.array(training_set[0])
    # y_data = np.array(training_set[1])
    training_loss = []

    ## initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())
        
    np.random.seed(random_seed) # for shuflling in batch_generator
    for epoch in range(1, epochs+1):



        # batch_gen = batch_generator(
        #                 X_train_centered, y_train, batch_size,
        #                 shuffle=shuffle)
        

        # def batch_generator(X, y, batch_size=64, 
        #             shuffle=False, random_seed=None):

        idx = np.arange(y_train.shape[0])
        if shuffle:
            rng = np.random.RandomState(random_seed)
            rng.shuffle(idx)
            X_train_centered = X_train_centered[idx]
            y_train = y_train[idx]

        count = 1
        avg_loss = 0.0
        for i in range(0, X_train_centered.shape[0], batch_size):
            feed = {'tf_x:0': X_train_centered[i:i+batch_size, :], 
                    'tf_y:0': y_train[i:i+batch_size], 
                    'fc_keep_prob:0': dropout}
            loss, _ = sess.run(
                    ['cross_entropy_loss:0', 'train_op'],
                    feed_dict=feed)
            avg_loss += loss

            if count % 100 == 0:
                print('epoch ', epoch, ' | batch ', count, ' finished')
            
            count += 1

        training_loss.append(avg_loss / (i+1))
        print('Epoch %02d Training Avg. Loss: %7.3f' % (epoch, avg_loss))
        if X_valid_centered is not None:
            feed = {'tf_x:0': X_valid_centered,
                    'tf_y:0': y_valid,
                    'fc_keep_prob:0':1.0} 
            valid_acc = sess.run('accuracy:0', feed_dict=feed)
            print(' Validation Acc: %7.3f' % valid_acc)
        else:
            print()

    save(saver, sess, epoch=2)

del g



# g2 = tf.Graph()
# with g2.as_default():
#     tf.set_random_seed(random_seed)
    
#     build_cnn(norm[0], lamb[0])

#     ## saver:
#     saver = tf.train.Saver()

# with tf.Session(graph=g2) as sess:
#     load(saver, sess, epoch=2, path='./model/')
    
#     preds = predict(sess, X_test_centered, return_proba=False)

#     print('Test Accuracy: %.3f%%' % (100*np.sum(preds == y_test)/len(y_test)))    




# def load_mnist(path, kind='train'):
#     """Load MNIST data from `path`"""
#     labels_path = os.path.join(path,
#                                '%s-labels-idx1-ubyte'
#                                 % kind)
#     images_path = os.path.join(path,
#                                '%s-images-idx3-ubyte'
#                                % kind)

#     with open(labels_path, 'rb') as lbpath:
#         magic, n = struct.unpack('>II',
#                                  lbpath.read(8))
#         labels = np.fromfile(lbpath,
#                              dtype=np.uint8)

#     with open(images_path, 'rb') as imgpath:
#         magic, num, rows, cols = struct.unpack(">IIII",
#                                                imgpath.read(16))
#         images = np.fromfile(imgpath,
#                              dtype=np.uint8).reshape(len(labels), 784)

#     return images, labels


# def batch_generator(X, y, batch_size=64, 
#                     shuffle=False, random_seed=None):
    
#     idx = np.arange(y.shape[0])
    
#     if shuffle:
#         rng = np.random.RandomState(random_seed)
#         rng.shuffle(idx)
#         X = X[idx]
#         y = y[idx]
    
#     for i in range(0, X.shape[0], batch_size):
#         yield (X[i:i+batch_size, :], y[i:i+batch_size])        


# def conv_layer(input_tensor, name,
#                kernel_size, n_output_channels, 
#                padding_mode='SAME', strides=(1, 1, 1, 1)):
#     with tf.variable_scope(name):
#         ## get n_input_channels:
#         ##   input tensor shape: 
#         ##   [batch x width x height x channels_in]
#         input_shape = input_tensor.get_shape().as_list()
#         n_input_channels = input_shape[-1] 

#         weights_shape = (list(kernel_size) + 
#                          [n_input_channels, n_output_channels])

#         weights = tf.get_variable(name='_weights',
#                                   shape=weights_shape)
#         print(weights)
#         biases = tf.get_variable(name='_biases',
#                                  initializer=tf.zeros(
#                                      shape=[n_output_channels]))
#         print(biases)
#         conv = tf.nn.conv2d(input=input_tensor, 
#                             filter=weights,
#                             strides=strides, 
#                             padding=padding_mode)
#         print(conv)
#         conv = tf.nn.bias_add(conv, biases, 
#                               name='net_pre-activation')
#         print(conv)
#         conv = tf.nn.relu(conv, name='activation')
#         print(conv)
        
#         return conv



# def fc_layer(input_tensor, name, 
#              n_output_units, activation_fn=None):
#     with tf.variable_scope(name):
#         input_shape = input_tensor.get_shape().as_list()[1:]
#         n_input_units = np.prod(input_shape)
#         if len(input_shape) > 1:
#             input_tensor = tf.reshape(input_tensor, 
#                                       shape=(-1, n_input_units))

#         weights_shape = [n_input_units, n_output_units]

#         weights = tf.get_variable(name='_weights',
#                                   shape=weights_shape)
#         print(weights)
#         biases = tf.get_variable(name='_biases',
#                                  initializer=tf.zeros(
#                                      shape=[n_output_units]))
#         print(biases)
#         layer = tf.matmul(input_tensor, weights)
#         print(layer)
#         layer = tf.nn.bias_add(layer, biases,
#                               name='net_pre-activation')
#         print(layer)
#         if activation_fn is None:
#             return layer
        
#         layer = activation_fn(layer, name='activation')
#         print(layer)
#         return layer, weights



# def build_cnn(norm, lamb):
#     ## Placeholders for X and y:
#     tf_x = tf.placeholder(tf.float32, shape=[None, 784],
#                           name='tf_x')
#     tf_y = tf.placeholder(tf.int32, shape=[None],
#                           name='tf_y')

#     # reshape x to a 4D tensor: 
#     # [batchsize, width, height, 1]
#     tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1],
#                             name='tf_x_reshaped')
#     ## One-hot encoding:
#     tf_y_onehot = tf.one_hot(indices=tf_y, depth=10,
#                              dtype=tf.float32,
#                              name='tf_y_onehot')

#     ## 1st layer: Conv_1
#     print('\nBuilding 1st layer: ')
#     h1 = conv_layer(tf_x_image, name='conv_1',
#                     kernel_size=(5, 5), 
#                     padding_mode='VALID',
#                     n_output_channels=32)
#     ## MaxPooling
#     h1_pool = tf.nn.max_pool(h1, 
#                              ksize=[1, 2, 2, 1],
#                              strides=[1, 2, 2, 1], 
#                              padding='SAME')
#     ## 2n layer: Conv_2
#     print('\nBuilding 2nd layer: ')
#     h2 = conv_layer(h1_pool, name='conv_2', 
#                     kernel_size=(5,5), 
#                     padding_mode='VALID',
#                     n_output_channels=64)
#     ## MaxPooling 
#     h2_pool = tf.nn.max_pool(h2, 
#                              ksize=[1, 2, 2, 1],
#                              strides=[1, 2, 2, 1], 
#                              padding='SAME')

#     ## 3rd layer: Fully Connected
#     print('\nBuilding 3rd layer:')
#     h3, h3_weights = fc_layer(h2_pool, name='fc_3',
#                   n_output_units=1024, 
#                   activation_fn=tf.nn.relu)

#     ## Dropout
#     keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
#     h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, 
#                             name='dropout_layer')

#     ## 4th layer: Fully Connected (linear activation)
#     print('\nBuilding 4th layer:')
#     h4 = fc_layer(h3_drop, name='fc_4',
#                   n_output_units=10, 
#                   activation_fn=None)

#     ## Prediction
#     predictions = {
#         'probabilities' : tf.nn.softmax(h4, name='probabilities'),
#         'labels' : tf.cast(tf.argmax(h4, axis=1), tf.int32,
#                            name='labels')
#     }
    
#     ## Visualize the graph with TensorBoard:

#     ## Loss Function and Optimization
#     cross_entropy_loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(
#             logits=h4, labels=tf_y_onehot),
#         name='cross_entropy_loss')


#     if norm == 'L1':
#         L1_norm = tf.norm(h3_weights, ord='fro', axis=(0,1))

#         ## Optimizer:
#         optimizer = tf.train.AdamOptimizer(learning_rate)
#         optimizer = optimizer.minimize(cross_entropy_loss+lamb*L1_norm,
#                                     name='train_op')

#     elif norm == 'L2':
#         L2_norm = tf.norm(h3_weights, ord='euclidean', axis=(0,1))

#         ## Optimizer:
#         optimizer = tf.train.AdamOptimizer(learning_rate)
#         optimizer = optimizer.minimize(cross_entropy_loss+lamb*L2_norm,
#                                     name='train_op')



#     ## Computing the prediction accuracy
#     correct_predictions = tf.equal(
#         predictions['labels'], 
#         tf_y, name='correct_preds')

#     accuracy = tf.reduce_mean(
#         tf.cast(correct_predictions, tf.float32),
#         name='accuracy')

# def save(saver, sess, epoch, path='./model/'):
#     if not os.path.isdir(path):
#         os.makedirs(path)
#     print('Saving model in %s' % path)
#     saver.save(sess, os.path.jo3in(path,'cnn-model.ckpt'),
#                global_step=epoch)

    
# def load(saver, sess, path, epoch):
#     print('Loading model from %s' % path)
#     saver.restore(sess, os.path.join(
#             path, 'cnn-model.ckpt-%d' % epoch))

    
# def train(sess, training_set, batch_size, validation_set=None,
#           initialize=True, epochs=20, shuffle=True,
#           dropout=0.5, random_seed=None):

#     X_data = np.array(training_set[0])
#     y_data = np.array(training_set[1])
#     training_loss = []

#     ## initialize variables
#     if initialize:
#         sess.run(tf.global_variables_initializer())
        
#     np.random.seed(random_seed) # for shuflling in batch_generator
#     for epoch in range(1, epochs+1):
#         batch_gen = batch_generator(
#                         X_data, y_data, batch_size,
#                         shuffle=shuffle)
#         avg_loss = 0.0
#         for i,(batch_x,batch_y) in enumerate(batch_gen):
#             feed = {'tf_x:0': batch_x, 
#                     'tf_y:0': batch_y, 
#                     'fc_keep_prob:0': dropout}
#             loss, _ = sess.run(
#                     ['cross_entropy_loss:0', 'train_op'],
#                     feed_dict=feed)
#             avg_loss += loss

#         training_loss.append(avg_loss / (i+1))
#         print('Epoch %02d Training Avg. Loss: %7.3f' % (epoch, avg_loss))
#         if validation_set is not None:
#             feed = {'tf_x:0': validation_set[0],
#                     'tf_y:0': validation_set[1],
#                     'fc_keep_prob:0':1.0} 
#             valid_acc = sess.run('accuracy:0', feed_dict=feed)
#             print(' Validation Acc: %7.3f' % valid_acc)
#         else:
#             print()

            
# def predict(sess, X_test, return_proba=False):
#     feed = {'tf_x:0': X_test, 
#             'fc_keep_prob:0': 1.0}
#     if return_proba:
#         return sess.run('probabilities:0', feed_dict=feed)
#     else:
#         return sess.run('labels:0', feed_dict=feed)


# keep_prob_param = [0.1, 0.5, 1.0]
# norm = ['L1', 'L2']
# lamb = [0, 0.1, 0.001]
# mini_batch = [1, 4, 16, 64, 128, 1024, 55000]



# if (sys.version_info > (3, 0)):
#     writemode = 'wb'
# else:
#     writemode = 'w'

# zipped_mnist = [f for f in os.listdir('./')
#                 if f.endswith('ubyte.gz')]
# for z in zipped_mnist:
#     with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
#         outfile.write(decompressed.read())

# X_data, y_data = load_mnist('./', kind='train')
# print('Rows: %d,  Columns: %d' % (X_data.shape[0], X_data.shape[1]))
# X_test, y_test = load_mnist('./', kind='t10k')
# print('Rows: %d,  Columns: %d' % (X_test.shape[0], X_test.shape[1]))

# X_train, y_train = X_data[:55000,:], y_data[:55000]
# X_valid, y_valid = X_data[55000:,:], y_data[55000:]

# print('Training:   ', X_train.shape, y_train.shape)
# print('Validation: ', X_valid.shape, y_valid.shape)
# print('Test Set:   ', X_test.shape, y_test.shape)


# mean_vals = np.mean(X_train, axis=0)
# std_val = np.std(X_train)

# X_train_centered = (X_train - mean_vals)/std_val
# X_valid_centered = X_valid - mean_vals
# X_test_centered = (X_test - mean_vals)/std_val

# print_('sys.getsizeof(X_data)')
# print_('sys.getsizeof(y_data)')
# print_('sys.getsizeof(X_train)')
# print_('sys.getsizeof(X_valid)')
# print_('sys.getsizeof(X_test)')
# print_('sys.getsizeof(X_train_centered)')
# print_('sys.getsizeof(X_valid_centered)')
# print_('sys.getsizeof(X_test_centered)')

# del X_data, y_data, X_train, X_valid, X_test



# # for i in keep_prob_param:
# #     for j in norm:
# #         for k in lamb:
# #             for l in mini_batch:
# i = keep_prob_param[0]
# j = norm[0]
# k = lamb[0]
# l = mini_batch[0]

# g = tf.Graph()
# with g.as_default():
#     tf.set_random_seed(random_seed)
    
#     build_cnn(j, k)
    
#     saver = tf.train.Saver()
# with tf.Session(graph=g) as sess:
#     train(sess, training_set=(X_train_centered, y_train), batch_size=l,
#         validation_set=(X_valid_centered, y_valid), initialize=True, random_seed=123, dropout=i, epochs=2)
#     save(saver, sess, epoch=2)

# del g


# g2 = tf.Graph()
# with g2.as_default():
#     tf.set_random_seed(random_seed)
    
#     build_cnn(norm[0], lamb[0])

#     ## saver:
#     saver = tf.train.Saver()

# ## create a new session 
# ## and restore the model
# with tf.Session(graph=g2) as sess:
#     load(saver, sess, epoch=20, path='./model/')
    
#     preds = predict(sess, X_test_centered, return_proba=False)

#     print('Test Accuracy: %.3f%%' % (100*np.sum(preds == y_test)/len(y_test)))    