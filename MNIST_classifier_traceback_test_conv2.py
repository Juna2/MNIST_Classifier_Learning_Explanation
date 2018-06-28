import matplotlib.pyplot as plt
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
    print(name, ' : \n', eval(name))


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
                             padding='SAME',
                             name='h2_pool')

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


def predict(sess, X_test, return_proba=False):
    feed = {'tf_x:0': X_test, 
            'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)

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

# z = input('pass 1 ?')

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

# z = input('pass 2 ?')

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

# z = input('pass 3 ?')

X_train, y_train = X_data[:55000,:], y_data[:55000]
X_valid, y_valid = X_data[55000:,:], y_data[55000:]
X_test, y_test = X_test[:1,:], y_test[:1]

print('Training:   ', X_train.shape, y_train.shape)
print('Validation: ', X_valid.shape, y_valid.shape)
print('Test Set:   ', X_test.shape, y_test.shape)


# z = input('pass 4 ?')

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_valid_centered = X_valid - mean_vals
X_test_centered = (X_test - mean_vals)/std_val
X_test_img = X_test[:1,:]

# z = input('pass 5 ?')

print_('sys.getsizeof(X_data)')
print_('sys.getsizeof(y_data)')
print_('sys.getsizeof(X_train)')
print_('sys.getsizeof(X_valid)')
print_('sys.getsizeof(X_test)')
print_('sys.getsizeof(X_train_centered)')
print_('sys.getsizeof(X_valid_centered)')
print_('sys.getsizeof(X_test_centered)')

del X_data, y_data, X_train, X_valid, X_test

# z = input('pass 6 ?')


# for i in keep_prob_param:
#     for j in norm:
#         for k in lamb:
#             for l in mini_batch:
# dropout = keep_prob_param[0]
# j = norm[0]
# k = lamb[0]
# batch_size = mini_batch[0]

# random_seed = 123
# initialize = True
# epochs = 2
# shuffle = 'shuffle'



g2 = tf.Graph()
with g2.as_default():
    tf.set_random_seed(random_seed)

    # z = input('pass 7 ?')
    
    build_cnn(norm[0], lamb[0])

    # z = input('pass 8_1 ?')

    ## saver:
    saver = tf.train.Saver()

    # z = input('pass 8_2 ?')

with tf.Session(graph=g2) as sess:

    # z = input('pass 8_3 ?')

    load(saver, sess, epoch=2, path='./model/')

    # z = input('pass 9 ?')

    return_proba = False

    feed = {'tf_x:0': X_test_centered, 
            'fc_keep_prob:0': 1.0}
    if return_proba:
        preds = sess.run('probabilities:0', feed_dict=feed)
    else:
        conv_2_result, fc_3_w, h2_pool, fc_3_b, fc_4_w, fc_3_result, fc_4_b, fc_4_result, preds = \
        sess.run(['conv_2/activation:0', 'fc_3/_weights:0', 'h2_pool:0', 'fc_3/_biases:0', 'fc_4/_weights:0', 'fc_3/activation:0', 'fc_4/_biases:0', 'fc_4/net_pre-activation:0','labels:0'], feed_dict=feed)
        # preds = sess.run('labels:0', feed_dict=feed)

    # z = input('pass 10 ?')

    img = np.resize(X_test_img * 1/255, (28,28))
    img = np.stack([img, img, img], axis=2)

    # pool_img = np.resize(h2_pool * 1/255, (4,4))
    # pool_img = np.stack([h2_pool, h2_pool, h2_pool], axis=2)
    
    # plt.figure(0)
    # plt.imshow(img)
    # plt.figure(1)
    '''Show h2 pooling layer's result'''
    # fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(4, 4))
    # for i in range(64):        
    #     h2_pool_tmp = np.array(h2_pool[0,:,:,i])
    #     # print_('h2_pool')
    #     pool_img = np.stack([h2_pool_tmp, h2_pool_tmp, h2_pool_tmp], axis=2)
    #     # row = i/8
    #     # column = i%8
    #     ax[int(i/8), int(i%8)].imshow(pool_img)
    
    
    # z = input('pass 11 ?')


    
    # mul_full = np.dot(fc_3_result, fc_4_w)
    # mul = np.dot(fc_3_result, fc_4_w[:,7])
    # # print_('fc_3_result.shape')
    # # print_('np.array([fc_4_w[:,7]]).shape')
    # mul_member = fc_3_result * np.array([fc_4_w[:,7]])
    # mul_member = mul_member[0,:]
    # sorted_index = np.argsort(mul_member)[::-1]
    # sum_1 = 0
    # sum_2 = 0
    # check_point = 0
    # for i in list(mul_member[sorted_index]):
    #     if i > 0:
    #         sum_1 += i
    # for index, i in enumerate(list(mul_member[sorted_index])):
    #     if i > 0:
    #         sum_2 += i
    #         if sum_2 / sum_1 > 0.5:
    #             check_point = index
    #             break
    # print_('sum_1')
    # print_('sum_2')                
    # print_('check_point')
    # activated_node = sorted_index[np.array(range(check_point))]
    # print_('activated_node')



    def activated_nodes(input_node, weights, range=0.5):
        mul = np.dot(input_node, weights)
        # print_('fc_3_result.shape')
        # print_('np.array([fc_4_w[:,7]]).shape')
        mul_member = input_node * np.array([weights])
        mul_member = mul_member[0,:]
        sorted_index = np.argsort(mul_member)[::-1]
        sum_1 = 0
        sum_2 = 0
        check_point = 0
        for i in list(mul_member[sorted_index]):
            if i > 0:
                sum_1 += i
        for index, i in enumerate(list(mul_member[sorted_index])):
            if i > 0:
                sum_2 += i
                if sum_2 / sum_1 > 0.5:
                    check_point = index
                    break
        # print('sum_1 :\n', sum_1)
        # print('sum_2 :\n', sum_2)                
        # print('check_point : \n', check_point)
        nodes = sorted_index[np.arange(check_point)]
        return nodes
    
    # def show(all_nodes, activated_nodes):
    #     '''Show h2 pooling layer's activated pixel'''
    #     fig, ax_act_h2_pool_1 = plt.subplots(nrows=8, ncols=8, figsize=(4, 4))
    #     channels = activated_nodes%(64*4)%64
    #     for i in range(64):
    #         h2_pool_tmp = np.array(all_nodes[0,:,:,i])
    #         pool_img = np.stack([h2_pool_tmp, h2_pool_tmp, h2_pool_tmp], axis=2)


    #         '''Make activated pixel red'''        
    #         for index, which_channel in enumerate(channels):
    #             if which_channel == i:
    #                 row = int(activated_nodes[index]%(64*4)/64)
    #                 # row = int(activated_node_h2_pool_1[index]%16/4)
    #                 column = int(activated_nodes[index]/(64*4))
    #                 # column = (activated_node_h2_pool_1[index]%16)%4
    #                 print_('row')
    #                 print_('column')
    #                 pool_img[row, column, 0] = 1
    #                 pool_img[row, column, 1] = 0
    #                 pool_img[row, column, 2] = 0

    #         ax_act_h2_pool_1[int(i/8), int(i%8)].imshow(pool_img)

    #     plt.show()

    activated_node_fc_3 = activated_nodes(fc_3_result, fc_4_w[:,7])
    print_('activated_node_fc_3')

    plot_or_not = True
    activated_node_h2_pool_all = np.array([[]])


    for num in range(0, 78):
        
        # z = input('pass 12 ?')

        h2_pool_resize = np.resize(h2_pool[0,:,:,:], (1,1024))

        activated_node_h2_pool = \
            activated_nodes(h2_pool_resize, fc_3_w[:,activated_node_fc_3[num]]) ####

        activated_node_h2_pool_all = np.c_[activated_node_h2_pool_all, [activated_node_h2_pool]]

    activated_node_h2_pool_all = np.unique(activated_node_h2_pool_all)
    print_('activated_node_h2_pool_all')


    column = np.array(activated_node_h2_pool_all/(64*4), dtype="i")
    row = np.array(activated_node_h2_pool_all%(64*4)/64, dtype="i")
    channels = np.array(activated_node_h2_pool_all%(64*4)%64, dtype="i")

    print_('column')
    print_('row')
    print_('channels')
    activated_node_h2_pool_all_3d = np.stack([column, row, channels], axis=1)
    print_('activated_node_h2_pool_all_3d')
    print_('activated_node_h2_pool_all_3d.shape')
    print_('activated_node_h2_pool_all_3d[:,2]')

    fig, ax_conv2 = plt.subplots(nrows=8, ncols=8, figsize=(15, 15))
    fig.tight_layout()

    for i in range(64):
        conv2_tmp = np.array(conv_2_result[0,:,:,i])
        conv2_img = np.stack([conv2_tmp, conv2_tmp, conv2_tmp], axis=2)

        for index, channel in enumerate(activated_node_h2_pool_all_3d[:,2]):
            if channel == i:
                column = activated_node_h2_pool_all_3d[index,0]
                row = activated_node_h2_pool_all_3d[index,1]

                pixel_4x4 = [float(conv2_tmp[2*column, 2*row]), \
                            float(conv2_tmp[2*column+1, 2*row]), \
                            float(conv2_tmp[2*column, 2*row+1]), \
                            float(conv2_tmp[2*column+1, 2*row+1])]
                maximum_value = max(pixel_4x4)                           
                maximum_index = pixel_4x4.index(maximum_value)

                if maximum_index == 0:
                    column = 2*column
                    row = 2*row
                    conv2_img[column, row, 0] = 1
                    conv2_img[column, row, 1] = 0
                    conv2_img[column, row, 2] = 0
                elif maximum_index == 1:
                    column = 2*column+1
                    row = 2*row
                    conv2_img[column, row, 0] = 1
                    conv2_img[column, row, 1] = 0
                    conv2_img[column, row, 2] = 0
                elif maximum_index == 2:
                    column = 2*column
                    row = 2*row+1
                    conv2_img[column, row, 0] = 1
                    conv2_img[column, row, 1] = 0
                    conv2_img[column, row, 2] = 0
                elif maximum_index == 3:
                    column = 2*column+1
                    row = 2*row+1
                    conv2_img[column, row, 0] = 1
                    conv2_img[column, row, 1] = 0
                    conv2_img[column, row, 2] = 0

        ax_conv2[int(i/8), int(i%8)].imshow(conv2_img)

    plt.show()
    filename = 'conv2_result .png'
    print(filename, ' saved')
    plt.savefig(filename)            

        





        # '''Show h2 pooling layer's activated pixel'''  
        # if plot_or_not == True: 
        #     fig, ax_act_h2_pool_1 = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
        #     fig.tight_layout()
        
        # channels = activated_node_h2_pool%(64*4)%64
        # for i in range(64):
        #     h2_pool_tmp = np.array(h2_pool[0,:,:,i])
        #     if plot_or_not == True:
        #         pool_img = np.stack([h2_pool_tmp, h2_pool_tmp, h2_pool_tmp], axis=2)


        #     '''Make activated pixel red'''        
        #     for index, which_channel in enumerate(channels):
        #         if which_channel == i:
        #             row = int(activated_node_h2_pool[index]%(64*4)/64)
        #             column = int(activated_node_h2_pool[index]/(64*4))
        #             print_('row')
        #             print_('column')
        #             if plot_or_not == True:
        #                 pool_img[column, row, 0] = 1
        #                 pool_img[column, row, 1] = 0
        #                 pool_img[column, row, 2] = 0
        #     if plot_or_not == True:
        #         ax_act_h2_pool_1[int(i/8), int(i%8)].imshow(pool_img)
            

        # if plot_or_not == True:
        #     filename = 'act_h2_pool_' + str(num+1) + '.png'
        #     print(filename, ' saved')
        #     plt.savefig(filename)












        # '''Show h2 pooling layer's activated pixel'''  
        # if plot_or_not == True: 
        #     fig, ax_act_h2_pool_1 = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
        #     fig.tight_layout()
        
        # channels = activated_node_h2_pool%(64*4)%64
        # for i in range(64):
        #     h2_pool_tmp = np.array(h2_pool[0,:,:,i])
        #     if plot_or_not == True:
        #         pool_img = np.stack([h2_pool_tmp, h2_pool_tmp, h2_pool_tmp], axis=2)


        #     '''Make activated pixel red'''        
        #     for index, which_channel in enumerate(channels):
        #         if which_channel == i:
        #             row = int(activated_node_h2_pool[index]%(64*4)/64)
        #             column = int(activated_node_h2_pool[index]/(64*4))
        #             print_('row')
        #             print_('column')
        #             if plot_or_not == True:
        #                 pool_img[column, row, 0] = 1
        #                 pool_img[column, row, 1] = 0
        #                 pool_img[column, row, 2] = 0
        #     if plot_or_not == True:
        #         ax_act_h2_pool_1[int(i/8), int(i%8)].imshow(pool_img)
            

        # if plot_or_not == True:
        #     filename = 'act_h2_pool_' + str(num+1) + '.png'
        #     print(filename, ' saved')
        #     plt.savefig(filename)





    # activated_node_h2_pool_all = np.sort(activated_node_h2_pool_all)
    # print_('activated_node_h2_pool_all')
    # activated_node_h2_pool_all = np.unique(activated_node_h2_pool_all)
    # print_('activated_node_h2_pool_all')
    # print_('h2_pool_resize.shape')
    # print('h2_pool[activated_node_h2_pool_all] : \n', h2_pool_resize[0,:][np.array(activated_node_h2_pool_all, dtype="i")])
    
    # print_('conv_2_result.shape')
    # for i in range(64):
    #     conv_2_result_channel = np.array(conv_2_result[0,:,:,i])
    #     h2_pool_tmp = np.array(h2_pool[0,:,:,i])
    #     h2_pool_resize = np.resize(h2_pool[0,:,:,:], (1,1024))
    #     conv_2_result_channel[]
    #     for j in range(int(8/2)**2)
            
    #         for l in conv_2_result_channel[]
    #         h2_pool_tmp[0,0]


    # # z = input('pass 14 ?')

    # np.set_printoptions(threshold=np.inf)
    # with open("/home/juna/Documents/Projects/cnn_trace_back/info/info.txt", "w") as f:
    #     for i in ['h2_pool', 'fc_4_w[:,6]', 'fc_3_result', 'mul_full', 'mul_member', 'mul', 'fc_4_b', 'fc_4_result']:
    #         f.write(i + ' :\n' + str(eval(i)) + '\n\n')
    # np.set_printoptions(threshold=1000)

    # print('Test Accuracy: %.3f%%' % (100*np.sum(preds == y_test)/len(y_test)))    


    # z = input('pass 15 ?')


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