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
                             padding='SAME',
                             name='h1_pool')
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
# X_test, y_test = X_test[:1,:], y_test[:1] #@@@@
number = 30
X_test, y_test = X_test[number:number+1,:], y_test[number:number+1]

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

# print_('sys.getsizeof(X_data)')
# print_('sys.getsizeof(y_data)')
# print_('sys.getsizeof(X_train)')
# print_('sys.getsizeof(X_valid)')
# print_('sys.getsizeof(X_test)')
# print_('sys.getsizeof(X_train_centered)')
# print_('sys.getsizeof(X_valid_centered)')
# print_('sys.getsizeof(X_test_centered)')

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
        image, Conv1Kernel, Conv1Bias, conv_1_result, result_of_h1_pool, Conv2Kernel, Conv2Bias, conv_2_result, fc_3_w, h2_pool, fc_3_b, fc_4_w, fc_3_result, fc_4_b, fc_4_result, preds = \
        sess.run(['tf_x:0', 'conv_1/_weights:0', 'conv_1/_biases:0', 'conv_1/activation:0','h1_pool:0', 'conv_2/_weights:0', 'conv_2/_biases:0', 'conv_2/activation:0', 'fc_3/_weights:0', 'h2_pool:0', 'fc_3/_biases:0', 'fc_4/_weights:0', 'fc_3/activation:0', 'fc_4/_biases:0', 'fc_4/net_pre-activation:0','labels:0'], feed_dict=feed)
        # preds = sess.run('labels:0', feed_dict=feed)

    # print_('Conv2Kernel.shape')
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



    # def activated_nodes(input_node, weights, range=0.5):
    #     # mul = np.dot(input_node, weights)
    #     # print_('fc_3_result.shape')
    #     # print_('np.array([fc_4_w[:,7]]).shape')
    #     mul_member = input_node * weights
    #     mul_member = mul_member[0,:]
    #     sorted_index = np.argsort(mul_member)[::-1]
    #     sum_1 = 0
    #     sum_2 = 0
    #     check_point = 0
    #     for i in list(mul_member[sorted_index]):
    #         if i > 0:
    #             sum_1 += i
    #     for index, i in enumerate(list(mul_member[sorted_index])):
    #         if i > 0:
    #             sum_2 += i
    #             if sum_2 / sum_1 > 0.5:
    #                 check_point = index
    #                 break
    #     # print('sum_1 :\n', sum_1)
    #     # print('sum_2 :\n', sum_2)                
    #     # print('check_point : \n', check_point)
    #     nodes = sorted_index[np.arange(check_point)]
    #     return nodes

    # def activated_nodes(input_node, weights, bias, FormalRedValue, scope=0.999):
    #     # print_('fc_3_result.shape')
    #     # print_('np.array([fc_4_w[:,7]]).shape')
    #     mul_member = input_node * weights + bias
    #     mul_member = mul_member[0,:]
    #     sorted_index = np.argsort(mul_member)[::-1]
    #     sum_1 = 0
    #     sum_2 = 0
    #     check_point = 0
    #     SortedMulMember = mul_member[sorted_index]
    #     for i in list(SortedMulMember):
    #         if i > 0:
    #             sum_1 += i
    #     for index, i in enumerate(list(SortedMulMember)):
    #         if i > 0:
    #             sum_2 += i
    #             if sum_2 / sum_1 > scope:
    #                 check_point = index
    #                 break
    #     # print('sum_2/sum_1!!!!!!!!!!!!!!!! :\n', sum_2/sum_1)                
    #     # print('check_point : \n', check_point)
    #     nodes = sorted_index[np.arange(check_point)]
    #     print('sum_1 :\n', sum_1)
    #     print('sum_2 :\n', sum_2)
    #     print('FormalRedValue :\n', FormalRedValue)
    #     RedValue = (SortedMulMember[np.arange(check_point)]/sum_2)*(FormalRedValue*(sum_2/sum_1))
    #     return nodes, RedValue

        # RedValue = SortedMulMember[np.arange(check_point)]/(FormalRedValue*(sum_2/sum_1))






    def activated_nodes(input_node, weights, bias, FormalRedValue, scope=0.999):
        # print_('fc_3_result.shape')
        # print_('np.array([fc_4_w[:,7]]).shape')
        mul_member = input_node * weights 
        mul_member = mul_member[0,:]
        sorted_index = np.argsort(mul_member)[::-1]
        sum_1 = 0
        sum_2 = 0
        check_point = 0
        SortedMulMember = mul_member[sorted_index]
        for i in list(SortedMulMember):
            if i > 0:
                sum_1 += i
        if sum_1 == 0:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            sys.exit(1) 
        for index, i in enumerate(list(SortedMulMember)):
            if i > 0:
                sum_2 += i
                if sum_2 / sum_1 > scope:
                    check_point = index
                    break
                    
        # print('check_point : \n', check_point)
        nodes = sorted_index[np.arange(check_point)]
        # RedValue = SortedMulMember[np.arange(check_point)]*FormalRedValue/sum_2
        RedValue = SortedMulMember[np.arange(check_point)]*FormalRedValue/sum_1
        # RedValue = (SortedMulMember[np.arange(check_point)]/sum_2)*(FormalRedValue*(sum_2/sum_1))

        return nodes, RedValue
        
        





    '''=============================fc3 <- fc4================================'''
    '''input : fc_4_result(1, 10)'''
    # print_('fc_3_result.shape') # (1, 1024)
    # print_('np.array([fc_4_w[:,7]]).shape') # (1, 1024)
    # print_('fc_4_b[7].shape') # Just a constant 
    PixelsToBeRedFc3, Fc3RedValue = activated_nodes(fc_3_result, np.array([fc_4_w[:,0]]), fc_4_b[7], 1) #@@@@
    '''======================================================================='''




    '''============================h2_pool <- fc3============================'''
    '''input : h2_pool(1, 4, 4, 64), PixelsToBeRedFc3(210,), fc_3_w(1024, 1024), fc_3_b(1024,)'''
    print('============================h2_pool <- fc3============================')
    AllRedPixelsPointsFromH2pool = np.array([[]])
    AllH2poolRedValue = np.array([[]])
    h2_pool_resize = np.resize(h2_pool[0,:,:,:], (1,1024))

    for num in range(PixelsToBeRedFc3.shape[0]):
        RedPixelsPointsFromH2pool, H2poolRedValue = \
            activated_nodes(h2_pool_resize, fc_3_w[:,PixelsToBeRedFc3[num]], fc_3_b[num], Fc3RedValue[num]) ####
        AllRedPixelsPointsFromH2pool = np.c_[AllRedPixelsPointsFromH2pool, [RedPixelsPointsFromH2pool]]
        AllH2poolRedValue = np.c_[AllH2poolRedValue, [H2poolRedValue]]


    np.set_printoptions(threshold=np.inf)
    # print_('AllRedPixelsPointsFromH2pool')
    # print_('h2_pool_resize[0,AllRedPixelsPointsFromH2pool[0,:].astype(int)]') # (1, 55923)
    
    np.set_printoptions(threshold=1000)



    del fc_3_w, fc_3_b, fc_4_w, fc_3_result, fc_4_b, fc_4_result, preds ####################

    # AllRedPixelsPointsFromH2poolUnique = np.unique(AllRedPixelsPointsFromH2pool)

    # column = np.array(AllRedPixelsPointsFromH2poolUnique/(64*4), dtype="i")
    # row = np.array(AllRedPixelsPointsFromH2poolUnique%(64*4)/64, dtype="i")
    # channels = np.array(AllRedPixelsPointsFromH2poolUnique%(64*4)%64, dtype="i")


    columns = np.array(AllRedPixelsPointsFromH2pool/(64*4), dtype="i")
    rows = np.array(AllRedPixelsPointsFromH2pool%(64*4)/64, dtype="i")
    channels = np.array(AllRedPixelsPointsFromH2pool%(64*4)%64, dtype="i")

    PixelsToBeRedH2pool = np.stack([columns, rows, channels, AllH2poolRedValue], axis=2)
    '''output : PixelsToBeRedH2pool(All red pixels' coord and channel)'''
    '''======================================================================='''
    print('=========================================================================')


    



    # print_('PixelsToBeRedH2pool.shape') # (1, 55923, 4)
    # print_('np.sum(PixelsToBeRedH2pool[0,:,3])') # 0.9913220198592825


    '''=========================Collect all same pixels' Red values========================='''
    '''input : PixelsToBeRedH2pool<(column, row, channel, redvalue)>, h2_pool<1x4x4x64>'''
    print('============================Collect all same pixel Red values============================')

    np.set_printoptions(threshold=np.inf)
    print_('PixelsToBeRedH2pool.shape')
    np.set_printoptions(threshold=1000)

    TempArray = np.zeros((h2_pool.shape))
    for i in PixelsToBeRedH2pool[0,:,:]:
        # print(i)
        TempArray[0,int(i[0]),int(i[1]),int(i[2])] += i[3]
    



    PixelsToBeRedH2pool = np.array([[]])
    for i in range(h2_pool.shape[1]):
        for j in range(h2_pool.shape[2]):
            for k in range(h2_pool.shape[3]):
                if TempArray[0,i,j,k] != 0:
                    PixelsToBeRedH2pool = np.c_[PixelsToBeRedH2pool,np.array([[i,j,k,TempArray[0,i,j,k]]])]

    PixelsToBeRedH2pool = np.resize(PixelsToBeRedH2pool, (int(PixelsToBeRedH2pool.shape[1]/4), 4))

    
    '''output : PixelsToBeRedH2pool<far_less_amount_of (column,row,channel,redvalue)>'''
    '''===================================================================================='''
    print('=================================================================================')




    # # # print_('PixelsToBeRedH2pool.shape') # (1024, 4)
    # # # print_('np.sum(PixelsToBeRedH2pool[:,3])') # 0.9913220198592825


    '''=======================conv2 <- h2_pool(unpooling)========================='''
    '''input : PixelsToBeRedH2pool(1024, 4), conv_2_result(1, 8, 8, 64) '''
    '''PixelsToBeRedH2pool contains all coords of red pixels in 4x4 channels'''
    print('============================conv2 <- h2_pool(unpooling)============================')

    # fig, ax_conv2 = plt.subplots(nrows=8, ncols=8, figsize=(15, 15))
    # fig.tight_layout()
    PixelsToBeRedConv2 = np.array([[]], dtype="i")
    
    for i in range(64):
        conv2_tmp = np.array(conv_2_result[0,:,:,i])
        # conv2_img = np.stack([conv2_tmp, conv2_tmp, conv2_tmp], axis=2)
        
        for index, channel in enumerate(PixelsToBeRedH2pool[:,2]):
            if channel == i:
                column = int(PixelsToBeRedH2pool[index,0])
                row = int(PixelsToBeRedH2pool[index,1])

                pixel_4x4 = [float(conv2_tmp[2*column, 2*row]), \
                            float(conv2_tmp[2*column+1, 2*row]), \
                            float(conv2_tmp[2*column, 2*row+1]), \
                            float(conv2_tmp[2*column+1, 2*row+1])]
                maximum_value = max(pixel_4x4)                           
                maximum_index = pixel_4x4.index(maximum_value)

                if maximum_index == 0:
                    column = 2*column
                    row = 2*row
                    # conv2_img[column, row, 0] = 1
                    # conv2_img[column, row, 1] = 0
                    # conv2_img[column, row, 2] = 0
                elif maximum_index == 1:
                    column = 2*column+1
                    row = 2*row
                    # conv2_img[column, row, 0] = 1
                    # conv2_img[column, row, 1] = 0
                    # conv2_img[column, row, 2] = 0
                elif maximum_index == 2:
                    column = 2*column
                    row = 2*row+1
                    # conv2_img[column, row, 0] = 1
                    # conv2_img[column, row, 1] = 0
                    # conv2_img[column, row, 2] = 0
                elif maximum_index == 3:
                    column = 2*column+1
                    row = 2*row+1
                    # conv2_img[column, row, 0] = 1
                    # conv2_img[column, row, 1] = 0
                    # conv2_img[column, row, 2] = 0

                # print_('PixelsToBeRedConv2.shape')
                # print_('np.array([[column, row, channel]]).shape')
                PixelsToBeRedConv2 = \
                            np.c_[PixelsToBeRedConv2, [[column, row, channel, PixelsToBeRedH2pool[index,3]]]]
                
             
        # ax_conv2[int(i/8), int(i%8)].imshow(conv2_img)
    PixelsToBeRedConv2 = np.resize(PixelsToBeRedConv2,(int(PixelsToBeRedConv2.shape[1]/4), 4))
    # print_('PixelsToBeRedConv2')

    # plt.show()
    # filename = 'conv2_result .png'
    # print(filename, ' saved')
    # plt.savefig(filename)     
    
    '''output : PixelsToBeRedConv2(All red pixel's coord and channel)'''     
    '''================================================================================'''
    print('============================================================================')





    # np.set_printoptions(threshold=np.inf)
    # print_('PixelsToBeRedConv2.shape')
    # print_('np.sum(PixelsToBeRedConv2[:,3])')
    # np.set_printoptions(threshold=1000)

    # '''===================================Show Red Pixels===================================='''
    # '''input : conv_2_result(1, 8, 8, 64) , RowNumber, ColNumber
    # PixelsToBeRedConv2(1024, 4)(All red pixel's coord and channel | column, row, channel, RedValue)'''
    # AllChannelsFromH2pool = conv_2_result[0,:,:,:]
    # fig, ax_conv_2_result = plt.subplots(nrows=8, ncols=8, figsize=(5, 5))
    # # fig.tight_layout()

    # for i in range(conv_2_result.shape[3]):
    #     OneChannelFromH2pool =  AllChannelsFromH2pool[:,:,i]    
    #     h2_img = np.stack([OneChannelFromH2pool, OneChannelFromH2pool, OneChannelFromH2pool], axis=2)
    #     for index, channel in enumerate(np.array(PixelsToBeRedConv2[:,2], dtype="i")):
    #         if channel == i:
    #             OnePixelToBeRed = np.array(PixelsToBeRedConv2[index], dtype="i")
    #             h2_img[OnePixelToBeRed[0], OnePixelToBeRed[1],0] = OnePixelToBeRed[3]
    #             h2_img[OnePixelToBeRed[0], OnePixelToBeRed[1],1] = 0
    #             h2_img[OnePixelToBeRed[0], OnePixelToBeRed[1],2] = 0
        
    #     ax_conv_2_result[int(i/8), int(i%8)].imshow(h2_img)

    # # plt.imshow(h2_img)
    # plt.show()
    # # filename = 'conv_2_result_result_act .png'
    # # print(filename, ' saved')
    # # plt.savefig(filename)     
    # # del h2_pool  
    # '''output : showing images'''
    # '''=================================================================================='''



    # print_('PixelsToBeRedConv2.shape') # (1024, 4)
    # print_('np.sum(PixelsToBeRedConv2[:,3])') # 0.9913220198592825




    '''===================================h1_pool <- conv2===================================='''
    '''input : result_of_h1_pool(1, 12, 12, 32), Conv2Kernel(5, 5, 32, 64), Conv2Bias(64,)
                                   PixelsToBeRedConv2(1024, 4)(column, row, channel, redvalue)'''
    print('============================h1_pool <- conv2============================')

    PixelsToBeRedH1pool = np.array([[]])

    AllChannelsFromH1pool = result_of_h1_pool[0,:,:,:]
    for i in range(Conv2Kernel.shape[3]): 
        for index, channel in enumerate(np.array(PixelsToBeRedConv2[:,2], dtype="i")):
            if channel == i:
                StartPointOfCutting5x5 = PixelsToBeRedConv2[index,:]
                Pixel5x5x32ValuefromH1pool = AllChannelsFromH1pool[int(StartPointOfCutting5x5[0]):int(StartPointOfCutting5x5[0])+4,\
                                                                   int(StartPointOfCutting5x5[1]):int(StartPointOfCutting5x5[1])+4, :]
                Kernel5x5x32Value = Conv2Kernel[:,:,:,i]
                
                Pixel1x800ValuefromH1pool = np.resize(Pixel5x5x32ValuefromH1pool, (1,800))
                Kernel1x800Value = np.resize(Kernel5x5x32Value, (1,800))
                np.set_printoptions(threshold=np.inf)
                # print_('Pixel1x800ValuefromH1pool*Kernel1x800Value+Conv2Bias[channel]')
                np.set_printoptions(threshold=1000)
                
                RedPixelsPoints1x800FromH1pool, H1poolRedValue = \
                    activated_nodes(Pixel1x800ValuefromH1pool, Kernel1x800Value, Conv2Bias[channel], StartPointOfCutting5x5[3])
                # print_('RedPixelsPoints1x800FromH1pool.shape')



                # RedPixelsPointsFromH2pool, H2poolRedValue = \
                #                     activated_nodes(h2_pool_resize, fc_3_w[:,PixelsToBeRedFc3[num]], Fc3RedValue[num]) ####
                # AllRedPixelsPointsFromH2pool = np.c_[AllRedPixelsPointsFromH2pool, [RedPixelsPointsFromH2pool]]
                # AllH2poolRedValue = np.c_[AllH2poolRedValue, [H2poolRedValue]]




                for index, RedPixelPoint1x800 in enumerate(RedPixelsPoints1x800FromH1pool):
                    # print_('StartPointOfCutting5x5[0]')
                    # RedPixelPointcolumn = StartPointOfCutting5x5[0]+RedPixelPoint1x25%5
                    # RedPixelPointrow = StartPointOfCutting5x5[1]+int(RedPixelPoint1x25/5)
                    RedPixelPointcolumn = StartPointOfCutting5x5[0]+int(RedPixelPoint1x800/(32*5))
                    RedPixelPointrow = StartPointOfCutting5x5[1]+int(RedPixelPoint1x800%(32*5)/32)
                    RedPixelPointChannel = RedPixelPoint1x800%(32*5)%32
                    PixelsToBeRedH1pool = np.c_[PixelsToBeRedH1pool, \
                                    [[RedPixelPointcolumn,RedPixelPointrow,RedPixelPointChannel,H1poolRedValue[index]]]]
    # print_('PixelsToBeRed.shape[1]')
    PixelsToBeRedH1pool = np.resize(PixelsToBeRedH1pool, (1,int(PixelsToBeRedH1pool.shape[1]/4),4))
    print_('PixelsToBeRedH1pool')


    '''output : showing images, PixelsToBeRedH1pool(All red pixel's coord and channel)'''
    '''=================================================================================='''
    print('===============================================================================')


    # print_('PixelsToBeRedH1pool.shape') # (1, 44094, 4)
    # print_('np.sum(PixelsToBeRedH1pool[0,:,3])') # 24.00001179611263


    '''=========================Collect all same pixels' Red values========================='''
    '''input : PixelsToBeRedH1pool(1, 44094, 4)(column, row, channel, redvalue), 
                                                            result_of_h1_pool(1, 12, 12, 32)'''
    np.set_printoptions(threshold=np.inf)
    print_('PixelsToBeRedH1pool.shape')
    np.set_printoptions(threshold=1000)

    TempArray = np.zeros((result_of_h1_pool.shape))
    for i in PixelsToBeRedH1pool[0,:,:]:
        # print(i)
        TempArray[0,int(i[0]),int(i[1]),int(i[2])] += i[3]
    

    PixelsToBeRedH1pool = np.array([[]])
    for i in range(result_of_h1_pool.shape[1]):
        for j in range(result_of_h1_pool.shape[2]):
            for k in range(result_of_h1_pool.shape[3]):
                if TempArray[0,i,j,k] != 0:
                    PixelsToBeRedH1pool = np.c_[PixelsToBeRedH1pool,np.array([[i,j,k,TempArray[0,i,j,k]]])]

    PixelsToBeRedH1pool = np.resize(PixelsToBeRedH1pool, (int(PixelsToBeRedH1pool.shape[1]/4), 4))

    
    '''output : PixelsToBeRedH1pool<far_less_amount_of (column,row,channel,redvalue)>'''
    '''===================================================================================='''

    # print_('PixelsToBeRedH1pool.shape') # (3919, 4)
    # print_('np.sum(PixelsToBeRedH1pool[:,3])') # 0.928163750246565







#     # print_('PixelsToBeRedH1pool.shape')
    # '''===================================Show Red Pixels===================================='''
    # '''input : result_of_h1_pool(1, 12, 12, 32) , RowNumber, ColNumber, **threshold**<if i <= 0.1:>
    # PixelsToBeRedH1pool(3919, 4)(All red pixel's coord and channel | column, row, channel, RedValue)'''
    # AllChannelsFromH2pool = result_of_h1_pool[0,:,:,:]
    # # plt.figure(0)
    # fig, ax_result_of_h1_pool = plt.subplots(nrows=4, ncols=8, figsize=(5, 5))
    # # fig.tight_layout()

    # '''Normalizing'''
    # print_('PixelsToBeRedH1pool[:,3].max()')
    # PixelsToBeRedH1pool_Nor = PixelsToBeRedH1pool
    # PixelsToBeRedH1pool_Nor[:,3] = (PixelsToBeRedH1pool[:,3] - PixelsToBeRedH1pool[:,3].min())/ \
    #                                (PixelsToBeRedH1pool[:,3].max() - PixelsToBeRedH1pool[:,3].min())


    # '''Remove under certain value'''
    # remove_list = np.array([[]])
    # for index, i in enumerate(PixelsToBeRedH1pool_Nor[:,3]):
    #     if i <= 0.1:
    #         remove_list = np.c_[remove_list, [index]]
    # PixelsToBeRedH1pool_Nor = np.delete(PixelsToBeRedH1pool_Nor, remove_list, axis=0)
    # PixelsToBeRedH1poolDeleted = np.delete(PixelsToBeRedH1pool, remove_list, axis=0)




    # '''Debugging'''
    # np.set_printoptions(threshold=np.inf)
    # # print_('PixelsToBeRedH1pool')
    # print_('np.sum(PixelsToBeRedH1poolDeleted[:,3])')
    # print_('PixelsToBeRedH1poolDeleted[:,3]')
    # np.set_printoptions(threshold=1000)


    # for i in range(result_of_h1_pool.shape[3]):
    #     OneChannelFromH2pool =  AllChannelsFromH2pool[:,:,i]    
    #     h2_img = np.stack([OneChannelFromH2pool, OneChannelFromH2pool, OneChannelFromH2pool], axis=2)
    #     for index, channel in enumerate(np.array(PixelsToBeRedH1pool_Nor[:,2], dtype="i")):
    #         if channel == i:
    #             OnePixelToBeRed = PixelsToBeRedH1pool_Nor[index]
    #             h2_img[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),0] = OnePixelToBeRed[3]
    #             h2_img[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),1] = 0
    #             h2_img[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),2] = 0
        
    #     ax_result_of_h1_pool[int(i/8), int(i%8)].imshow(h2_img, cmap="hot")

    # # plt.imshow(h2_img)
    # plt.show()
    # # filename = 'result_of_h1_pool_result_act .png'
    # # print(filename, ' saved')
    # # plt.savefig(filename)     
    # # del h2_pool  
    # '''output : showing images'''
    # '''====================================================================================='''

#     # # x = np.arange(0,PixelsToBeRedH1pool[:,3].shape[0])
#     # # y = PixelsToBeRedH1pool[:,3]
#     # # plt.figure(1)
#     # # plt.plot(x, y, 'r-')
#     # # plt.show()

    

    '''=======================conv1 <- h1_pool(unpooling)========================='''
    '''input : PixelsToBeRedH1pool(3919, 4), conv_1_result(1, 24, 24, 32), '''
    '''PixelsToBeRedH1pool contains all coords of red pixels in 4x4 channels'''
    print('=======================conv1 <- h1_pool(unpooling)=========================')


    # fig, ax_conv2 = plt.subplots(nrows=8, ncols=8, figsize=(15, 15))
    # fig.tight_layout()
    PixelsToBeRedConv1 = np.array([[]], dtype="i")
    
    print_('conv_1_result.shape')
    print_('PixelsToBeRedH1pool.shape')

    for i in range(conv_1_result.shape[3]):
        conv1_tmp = np.array(conv_1_result[0,:,:,i])
        # conv1_img = np.stack([conv1_tmp, conv1_tmp, conv1_tmp], axis=2)
        
        for index, channel in enumerate(PixelsToBeRedH1pool[:,2]):
            if channel == i:
                column = int(PixelsToBeRedH1pool[index,0])
                row = int(PixelsToBeRedH1pool[index,1])

                pixel_4x4 = [float(conv1_tmp[2*column, 2*row]), \
                            float(conv1_tmp[2*column+1, 2*row]), \
                            float(conv1_tmp[2*column, 2*row+1]), \
                            float(conv1_tmp[2*column+1, 2*row+1])]
                maximum_value = max(pixel_4x4)
                maximum_index = pixel_4x4.index(maximum_value)

                if maximum_index == 0:
                    column = 2*column
                    row = 2*row
                    # conv1_img[column, row, 0] = 1
                    # conv1_img[column, row, 1] = 0
                    # conv1_img[column, row, 2] = 0
                elif maximum_index == 1:
                    column = 2*column+1
                    row = 2*row
                    # conv1_img[column, row, 0] = 1
                    # conv1_img[column, row, 1] = 0
                    # conv1_img[column, row, 2] = 0
                elif maximum_index == 2:
                    column = 2*column
                    row = 2*row+1
                    # conv1_img[column, row, 0] = 1
                    # conv1_img[column, row, 1] = 0
                    # conv1_img[column, row, 2] = 0
                elif maximum_index == 3:
                    column = 2*column+1
                    row = 2*row+1
                    # conv1_img[column, row, 0] = 1
                    # conv1_img[column, row, 1] = 0
                    # conv1_img[column, row, 2] = 0

                # print_('PixelsToBeRedConv1.shape')
                # print_('np.array([[column, row, channel]]).shape')
                PixelsToBeRedConv1 = \
                            np.c_[PixelsToBeRedConv1, [[column, row, channel, PixelsToBeRedH1pool[index,3]]]]
                
             
        # ax_conv1[int(i/8), int(i%8)].imshow(conv1_img)
    PixelsToBeRedConv1 = np.resize(PixelsToBeRedConv1,(int(PixelsToBeRedConv1.shape[1]/4), 4))
    # print_('PixelsToBeRedConv1')

    # plt.show()
    # filename = 'conv1_result .png'
    # print(filename, ' saved')
    # plt.savefig(filename)     
    
    '''output : PixelsToBeRedConv1(All red pixel's coord and channel)'''     
    '''================================================================================'''
    print('==================================================================================')


    # print_('PixelsToBeRedConv1.shape') # (3919, 4)
    # print_('np.sum(PixelsToBeRedConv1[:,3])') # 0.928163750246565



    
#     # print_('PixelsToBeRedConv1[:,3]')
    # '''===================================Show Red Pixels===================================='''
    # '''input : conv_1_result(1, 24, 24, 32) , RowNumber, ColNumber, **threshold**<if i <= 0.1:>
    # PixelsToBeRedConv1(3919, 4)(All red pixel's coord and channel | column, row, channel, RedValue)'''
    # AllChannelsFromConv1 = conv_1_result[0,:,:,:]
    # # plt.figure(0)
    # fig, ax_conv_1_result = plt.subplots(nrows=4, ncols=8, figsize=(5, 5))
    # # fig.tight_layout()

    # '''Normalizing (from 0 to 1)'''
    # print_('PixelsToBeRedConv1[:,3].max()')
    # PixelsToBeRedConv1_Nor = np.empty_like(PixelsToBeRedConv1)
    # np.copyto(PixelsToBeRedConv1_Nor, PixelsToBeRedConv1)
    # PixelsToBeRedConv1_Nor[:,3] = (PixelsToBeRedConv1_Nor[:,3] - PixelsToBeRedConv1_Nor[:,3].min())/ \
    #                                (PixelsToBeRedConv1_Nor[:,3].max() - PixelsToBeRedConv1_Nor[:,3].min())
    # print_('PixelsToBeRedConv1[:,3].max()')


    # '''Remove under certain value'''
    # remove_list = np.array([[]])
    # for index, i in enumerate(PixelsToBeRedConv1_Nor[:,3]):
    #     if i <= 0.05:
    #         remove_list = np.c_[remove_list, [index]]
    # PixelsToBeRedConv1_Nor = np.delete(PixelsToBeRedConv1_Nor, remove_list, axis=0)
    # PixelsToBeRedConv1Deleted = np.delete(PixelsToBeRedConv1, remove_list, axis=0)


    # '''Debugging'''
    # np.set_printoptions(threshold=np.inf)
    # # print_('PixelsToBeRedConv1')
    # print_('np.sum(PixelsToBeRedConv1Deleted[:,3])')
    # print_('PixelsToBeRedConv1Deleted[:,3]')
    # np.set_printoptions(threshold=1000)


    # '''image plot'''
    # for i in range(conv_1_result.shape[3]):
    #     OneChannelFromConv1 =  AllChannelsFromConv1[:,:,i]    
    #     h2_img = np.stack([OneChannelFromConv1, OneChannelFromConv1, OneChannelFromConv1], axis=2)
    #     for index, channel in enumerate(np.array(PixelsToBeRedConv1_Nor[:,2], dtype="i")):
    #         if channel == i:
    #             OnePixelToBeRed = PixelsToBeRedConv1_Nor[index]
    #             h2_img[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),0] = OnePixelToBeRed[3] +0.5
    #             h2_img[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),1] = 0
    #             h2_img[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),2] = 0
        
    #     ax_conv_1_result[int(i/8), int(i%8)].imshow(h2_img)

    # # plt.imshow(h2_img)
    # plt.show()
    # # filename = 'conv_1_result_result_act .png'
    # # print(filename, ' saved')
    # # plt.savefig(filename)     
    # # del h2_pool  
    # '''output : showing images'''
    # '''====================================================================================='''




    '''===================================image <- conv1===================================='''
    '''input : image(1, 784), Conv1Kernel(5, 5, 1, 32), Conv1Bias(32,)
                                 PixelsToBeRedConv1(3919, 4)(column, row, channel, redvalue)'''
    print('===================================image <- conv1====================================')
    PixelsToBeRedImage = np.array([[]])

    Image2D = np.resize(image, (28,28))

    print_('image')

    for i in range(Conv1Kernel.shape[2]): 
        for index, channel in enumerate(np.array(PixelsToBeRedConv1[:,2], dtype="i")):
            if channel == i:
                StartPointOfCutting5x5 = PixelsToBeRedConv1[index,:]
                Pixel5x5ValuefromImage = Image2D[int(StartPointOfCutting5x5[0]):int(StartPointOfCutting5x5[0])+4,\
                                                 int(StartPointOfCutting5x5[1]):int(StartPointOfCutting5x5[1])+4]
                Kernel5x5Value = Conv1Kernel[:,:,i]
                
                length = Kernel5x5Value.shape[0] * Kernel5x5Value.shape[1]
                Pixel1x25ValuefromImage = np.resize(Pixel5x5ValuefromImage, (1,length))
                Kernel1x25Value = np.resize(Kernel5x5Value, (1,length))
                
                RedPixelsPointsFromImage1D, ImageRedValue = \
                    activated_nodes(Pixel1x25ValuefromImage, Kernel1x25Value, Conv1Bias[channel], StartPointOfCutting5x5[3])
                # print_('RedPixelsPointsFromImage1D.shape')

                # RedPixelsPointsFromH2pool, H2poolRedValue = \
                #                     activated_nodes(h2_pool_resize, fc_3_w[:,PixelsToBeRedFc3[num]], Fc3RedValue[num]) ####
                # AllRedPixelsPointsFromH2pool = np.c_[AllRedPixelsPointsFromH2pool, [RedPixelsPointsFromH2pool]]
                # AllH2poolRedValue = np.c_[AllH2poolRedValue, [H2poolRedValue]]

                for index, RedPixelPoint1D in enumerate(RedPixelsPointsFromImage1D):
                    # print_('StartPointOfCutting5x5[0]')
                    # RedPixelPointcolumn = StartPointOfCutting5x5[0]+RedPixelPoint1x25%5
                    # RedPixelPointrow = StartPointOfCutting5x5[1]+int(RedPixelPoint1x25/5)
                    RedPixelPointcolumn = StartPointOfCutting5x5[0]+RedPixelPoint1D%5
                    RedPixelPointrow = StartPointOfCutting5x5[1]+int(RedPixelPoint1D/5)
                    PixelsToBeRedImage = np.c_[PixelsToBeRedImage, \
                                    [[RedPixelPointcolumn,RedPixelPointrow,ImageRedValue[index]]]]
    # print_('PixelsToBeRed.shape[1]')
    PixelsToBeRedImage = np.resize(PixelsToBeRedImage, (1,int(PixelsToBeRedImage.shape[1]/3),3))
    print_('PixelsToBeRedImage')


    '''output : showing images, PixelsToBeRedImage(All red pixel's coord and channel)'''
    '''=================================================================================='''
    print('==================================================================================')


    print_('PixelsToBeRedImage.shape') # (1, 2430, 3)
    print_('np.sum(PixelsToBeRedImage[0,:,2])') # 6.000000505500566


    '''=========================Collect all same pixels' Red values========================='''
    print('=========================Collect all same pixels Red values=========================')
    '''input : PixelsToBeRedImage(1, 2430, 3)(column, row, redvalue), image(1, 784)'''
    np.set_printoptions(threshold=np.inf)
    print_('PixelsToBeRedImage.shape')
    np.set_printoptions(threshold=1000)

    # print_('image.shape')
    Image2DTmp = np.resize(image, (28,28))
    ImageTempArray = np.zeros((Image2DTmp.shape))
    for i in PixelsToBeRedImage[0,:,:]:
        # print(i)
        ImageTempArray[int(i[0]),int(i[1])] += i[2]
    

    PixelsToBeRedImage = np.array([[]])
    for i in range(Image2DTmp.shape[0]):
        for j in range(Image2DTmp.shape[1]):
            if ImageTempArray[i,j] != 0:
                PixelsToBeRedImage = np.c_[PixelsToBeRedImage,np.array([[i,j,ImageTempArray[i,j]]])]

    PixelsToBeRedImage = np.resize(PixelsToBeRedImage, (int(PixelsToBeRedImage.shape[1]/3), 3))

    
    '''output : PixelsToBeRedImage<far_less_amount_of (column,row,channel,redvalue)>'''
    '''===================================================================================='''
    print('==================================================================================')


    print_('PixelsToBeRedImage.shape') # (618, 3)
    print_('np.sum(PixelsToBeRedImage[:,2])') # 0.07562872974755308





#     # print_('PixelsToBeRedImage')
    '''===================================Show Red Pixels===================================='''
    print('===================================Show Red Pixels====================================')
    '''input : image(1, 784) , RowNumber, ColNumber, **threshold**<if i <= 0.1:>
    PixelsToBeRedImage(618, 3)(All red pixel's coord and channel | column, row, RedValue)'''
    Image2D = np.resize(image, (28,28))

    # plt.figure(0)
    # fig.tight_layout()

    '''Normalizing (from 0 to 1)'''
    print_('PixelsToBeRedImage[:,2].max()')
    PixelsToBeRedImage_Nor = np.empty_like(PixelsToBeRedImage)
    np.copyto(PixelsToBeRedImage_Nor, PixelsToBeRedImage)
    PixelsToBeRedImage_Nor[:,2] = (PixelsToBeRedImage_Nor[:,2] - PixelsToBeRedImage_Nor[:,2].min())/ \
                                   (PixelsToBeRedImage_Nor[:,2].max() - PixelsToBeRedImage_Nor[:,2].min())
    print_('PixelsToBeRedImage[:,2].max()')


    '''Remove under certain value'''
    remove_list = np.array([[]])
    for index, i in enumerate(PixelsToBeRedImage_Nor[:,2]):
        if i <= 0.6: #@@@@
            remove_list = np.c_[remove_list, [index]]
    PixelsToBeRedImage_Nor = np.delete(PixelsToBeRedImage_Nor, remove_list, axis=0)
    PixelsToBeRedImageDeleted = np.delete(PixelsToBeRedImage, remove_list, axis=0)


    '''Debugging'''
    np.set_printoptions(threshold=np.inf)
    # print_('PixelsToBeRedImage')
    print_('np.sum(PixelsToBeRedImageDeleted[:,2])')
    print_('PixelsToBeRedImageDeleted[:,2]')
    print_('PixelsToBeRedImage.shape')
    print_('np.sum(PixelsToBeRedImage[:,2])')
    np.set_printoptions(threshold=1000)


    '''image plot'''
    # print_('image[0]')
    image_norm = (image - image.min()) / (image.max() - image.min())
    Image2D = np.resize(image_norm, (28,28))
    FinalImage = np.stack([Image2D, Image2D, Image2D], axis=2)

    for OnePixelToBeRed in PixelsToBeRedImage_Nor:
        # print_('OnePixelToBeRed[2]')
        FinalImage[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),0] = OnePixelToBeRed[2]
        FinalImage[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),1] = 0
        FinalImage[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),2] = 0
    
    # plt.imshow(FinalImage)
    # plt.show()

    # np.set_printoptions(threshold=np.inf)
    # print_('FinalImage.shape')
    # # print_('FinalImage')
    # np.set_printoptions(threshold=1000)

    plt.imshow(FinalImage)
    # plt.show()
    filename = 'Final_image_No1_0.20 .png'
    print(filename, ' saved')
    plt.savefig(filename)     
    # # del h2_pool  
    # '''output : showing images'''
    # '''====================================================================================='''





























# #     # def activated_nodes(input_node, weights, FormalRedValue, scope=0.5):
# #     #     # mul = np.dot(input_node, weights)
# #     #     # print_('fc_3_result.shape')
# #     #     # print_('np.array([fc_4_w[:,7]]).shape')
# #     #     mul_member = input_node * weights
# #     #     mul_member = mul_member[0,:]
# #     #     sorted_index = np.argsort(mul_member)[::-1]
# #     #     SortedMulRedValue = mul_member[sorted_index]
# #     #     # print(type(SortedMulRedValue))
# #     #     sum_1 = 0
# #     #     sum_2 = 0
# #     #     check_point = 0

# #     #     for i in list(SortedMulRedValue):
# #     #         if i > 0:
# #     #             sum_1 += i
# #     #     for index, i in enumerate(list(SortedMulRedValue)):
# #     #         if i > 0:
# #     #             sum_2 += i
# #     #             if sum_2 / sum_1 > scope:
# #     #                 check_point = index
# #     #                 break
        
# #     #     # print('SortedMulRedValue.shape[1] : \n',SortedMulRedValue.shape)
# #     #     SortedMulRedValue[check_point:int(SortedMulRedValue.shape[0])+1] = 0
# #     #     SortedMulRedValue = SortedMulRedValue * FormalRedValue/sum_2

# #     #     for i in range(SortedMulRedValue.shape[0]):
# #     #         mul_member[sorted_index][i] = SortedMulRedValue[i]
# #     #     return mul_member





# #         # '''Show h2 pooling layer's activated pixel'''  
# #         # if plot_or_not == True: 
# #         #     fig, ax_act_h2_pool_1 = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
# #         #     fig.tight_layout()
        
# #         # channels = activated_node_h2_pool%(64*4)%64
# #         # for i in range(64):
# #         #     h2_pool_tmp = np.array(h2_pool[0,:,:,i])
# #         #     if plot_or_not == True:
# #         #         pool_img = np.stack([h2_pool_tmp, h2_pool_tmp, h2_pool_tmp], axis=2)


# #         #     '''Make activated pixel red'''        
# #         #     for index, which_channel in enumerate(channels):
# #         #         if which_channel == i:
# #         #             row = int(activated_node_h2_pool[index]%(64*4)/64)
# #         #             column = int(activated_node_h2_pool[index]/(64*4))
# #         #             print_('row')
# #         #             print_('column')
# #         #             if plot_or_not == True:
# #         #                 pool_img[column, row, 0] = 1
# #         #                 pool_img[column, row, 1] = 0
# #         #                 pool_img[column, row, 2] = 0
# #         #     if plot_or_not == True:
# #         #         ax_act_h2_pool_1[int(i/8), int(i%8)].imshow(pool_img)
            

# #         # if plot_or_not == True:
# #         #     filename = 'act_h2_pool_' + str(num+1) + '.png'
# #         #     print(filename, ' saved')
# #         #     plt.savefig(filename)












# #         # '''Show h2 pooling layer's activated pixel'''  
# #         # if plot_or_not == True: 
# #         #     fig, ax_act_h2_pool_1 = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
# #         #     fig.tight_layout()
        
# #         # channels = activated_node_h2_pool%(64*4)%64
# #         # for i in range(64):
# #         #     h2_pool_tmp = np.array(h2_pool[0,:,:,i])
# #         #     if plot_or_not == True:
# #         #         pool_img = np.stack([h2_pool_tmp, h2_pool_tmp, h2_pool_tmp], axis=2)


# #         #     '''Make activated pixel red'''        
# #         #     for index, which_channel in enumerate(channels):
# #         #         if which_channel == i:
# #         #             row = int(activated_node_h2_pool[index]%(64*4)/64)
# #         #             column = int(activated_node_h2_pool[index]/(64*4))
# #         #             print_('row')
# #         #             print_('column')
# #         #             if plot_or_not == True:
# #         #                 pool_img[column, row, 0] = 1
# #         #                 pool_img[column, row, 1] = 0
# #         #                 pool_img[column, row, 2] = 0
# #         #     if plot_or_not == True:
# #         #         ax_act_h2_pool_1[int(i/8), int(i%8)].imshow(pool_img)
            

# #         # if plot_or_not == True:
# #         #     filename = 'act_h2_pool_' + str(num+1) + '.png'
# #         #     print(filename, ' saved')
# #         #     plt.savefig(filename)





# #     # activated_node_h2_pool_all = np.sort(activated_node_h2_pool_all)
# #     # print_('activated_node_h2_pool_all')
# #     # activated_node_h2_pool_all = np.unique(activated_node_h2_pool_all)
# #     # print_('activated_node_h2_pool_all')
# #     # print_('h2_pool_resize.shape')
# #     # print('h2_pool[activated_node_h2_pool_all] : \n', h2_pool_resize[0,:][np.array(activated_node_h2_pool_all, dtype="i")])
    
# #     # print_('conv_2_result.shape')
# #     # for i in range(64):
# #     #     conv_2_result_channel = np.array(conv_2_result[0,:,:,i])
# #     #     h2_pool_tmp = np.array(h2_pool[0,:,:,i])
# #     #     h2_pool_resize = np.resize(h2_pool[0,:,:,:], (1,1024))
# #     #     conv_2_result_channel[]
# #     #     for j in range(int(8/2)**2)
            
# #     #         for l in conv_2_result_channel[]
# #     #         h2_pool_tmp[0,0]


# #     # # z = input('pass 14 ?')

# #     # np.set_printoptions(threshold=np.inf)
# #     # with open("/home/juna/Documents/Projects/cnn_trace_back/info/info.txt", "w") as f:
# #     #     for i in ['h2_pool', 'fc_4_w[:,6]', 'fc_3_result', 'mul_full', 'mul_member', 'mul', 'fc_4_b', 'fc_4_result']:
# #     #         f.write(i + ' :\n' + str(eval(i)) + '\n\n')
# #     # np.set_printoptions(threshold=1000)

# #     # print('Test Accuracy: %.3f%%' % (100*np.sum(preds == y_test)/len(y_test)))    


# #     # z = input('pass 15 ?')


# # # def load_mnist(path, kind='train'):
# # #     """Load MNIST data from `path`"""
# # #     labels_path = os.path.join(path,
# # #                                '%s-labels-idx1-ubyte'
# # #                                 % kind)
# # #     images_path = os.path.join(path,
# # #                                '%s-images-idx3-ubyte'
# # #                                % kind)

# # #     with open(labels_path, 'rb') as lbpath:
# # #         magic, n = struct.unpack('>II',
# # #                                  lbpath.read(8))
# # #         labels = np.fromfile(lbpath,
# # #                              dtype=np.uint8)

# # #     with open(images_path, 'rb') as imgpath:
# # #         magic, num, rows, cols = struct.unpack(">IIII",
# # #                                                imgpath.read(16))
# # #         images = np.fromfile(imgpath,
# # #                              dtype=np.uint8).reshape(len(labels), 784)

# # #     return images, labels


# # # def batch_generator(X, y, batch_size=64, 
# # #                     shuffle=False, random_seed=None):
    
# # #     idx = np.arange(y.shape[0])
    
# # #     if shuffle:
# # #         rng = np.random.RandomState(random_seed)
# # #         rng.shuffle(idx)
# # #         X = X[idx]
# # #         y = y[idx]
    
# # #     for i in range(0, X.shape[0], batch_size):
# # #         yield (X[i:i+batch_size, :], y[i:i+batch_size])        


# # # def conv_layer(input_tensor, name,
# # #                kernel_size, n_output_channels, 
# # #                padding_mode='SAME', strides=(1, 1, 1, 1)):
# # #     with tf.variable_scope(name):
# # #         ## get n_input_channels:
# # #         ##   input tensor shape: 
# # #         ##   [batch x width x height x channels_in]
# # #         input_shape = input_tensor.get_shape().as_list()
# # #         n_input_channels = input_shape[-1] 

# # #         weights_shape = (list(kernel_size) + 
# # #                          [n_input_channels, n_output_channels])

# # #         weights = tf.get_variable(name='_weights',
# # #                                   shape=weights_shape)
# # #         print(weights)
# # #         biases = tf.get_variable(name='_biases',
# # #                                  initializer=tf.zeros(
# # #                                      shape=[n_output_channels]))
# # #         print(biases)
# # #         conv = tf.nn.conv2d(input=input_tensor, 
# # #                             filter=weights,
# # #                             strides=strides, 
# # #                             padding=padding_mode)
# # #         print(conv)
# # #         conv = tf.nn.bias_add(conv, biases, 
# # #                               name='net_pre-activation')
# # #         print(conv)
# # #         conv = tf.nn.relu(conv, name='activation')
# # #         print(conv)
        
# # #         return conv



# # # def fc_layer(input_tensor, name, 
# # #              n_output_units, activation_fn=None):
# # #     with tf.variable_scope(name):
# # #         input_shape = input_tensor.get_shape().as_list()[1:]
# # #         n_input_units = np.prod(input_shape)
# # #         if len(input_shape) > 1:
# # #             input_tensor = tf.reshape(input_tensor, 
# # #                                       shape=(-1, n_input_units))

# # #         weights_shape = [n_input_units, n_output_units]

# # #         weights = tf.get_variable(name='_weights',
# # #                                   shape=weights_shape)
# # #         print(weights)
# # #         biases = tf.get_variable(name='_biases',
# # #                                  initializer=tf.zeros(
# # #                                      shape=[n_output_units]))
# # #         print(biases)
# # #         layer = tf.matmul(input_tensor, weights)
# # #         print(layer)
# # #         layer = tf.nn.bias_add(layer, biases,
# # #                               name='net_pre-activation')
# # #         print(layer)
# # #         if activation_fn is None:
# # #             return layer
        
# # #         layer = activation_fn(layer, name='activation')
# # #         print(layer)
# # #         return layer, weights



# # # def build_cnn(norm, lamb):
# # #     ## Placeholders for X and y:
# # #     tf_x = tf.placeholder(tf.float32, shape=[None, 784],
# # #                           name='tf_x')
# # #     tf_y = tf.placeholder(tf.int32, shape=[None],
# # #                           name='tf_y')

# # #     # reshape x to a 4D tensor: 
# # #     # [batchsize, width, height, 1]
# # #     tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1],
# # #                             name='tf_x_reshaped')
# # #     ## One-hot encoding:
# # #     tf_y_onehot = tf.one_hot(indices=tf_y, depth=10,
# # #                              dtype=tf.float32,
# # #                              name='tf_y_onehot')

# # #     ## 1st layer: Conv_1
# # #     print('\nBuilding 1st layer: ')
# # #     h1 = conv_layer(tf_x_image, name='conv_1',
# # #                     kernel_size=(5, 5), 
# # #                     padding_mode='VALID',
# # #                     n_output_channels=32)
# # #     ## MaxPooling
# # #     h1_pool = tf.nn.max_pool(h1, 
# # #                              ksize=[1, 2, 2, 1],
# # #                              strides=[1, 2, 2, 1], 
# # #                              padding='SAME')
# # #     ## 2n layer: Conv_2
# # #     print('\nBuilding 2nd layer: ')
# # #     h2 = conv_layer(h1_pool, name='conv_2', 
# # #                     kernel_size=(5,5), 
# # #                     padding_mode='VALID',
# # #                     n_output_channels=64)
# # #     ## MaxPooling 
# # #     h2_pool = tf.nn.max_pool(h2, 
# # #                              ksize=[1, 2, 2, 1],
# # #                              strides=[1, 2, 2, 1], 
# # #                              padding='SAME')

# # #     ## 3rd layer: Fully Connected
# # #     print('\nBuilding 3rd layer:')
# # #     h3, h3_weights = fc_layer(h2_pool, name='fc_3',
# # #                   n_output_units=1024, 
# # #                   activation_fn=tf.nn.relu)

# # #     ## Dropout
# # #     keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
# # #     h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, 
# # #                             name='dropout_layer')

# # #     ## 4th layer: Fully Connected (linear activation)
# # #     print('\nBuilding 4th layer:')
# # #     h4 = fc_layer(h3_drop, name='fc_4',
# # #                   n_output_units=10, 
# # #                   activation_fn=None)

# # #     ## Prediction
# # #     predictions = {
# # #         'probabilities' : tf.nn.softmax(h4, name='probabilities'),
# # #         'labels' : tf.cast(tf.argmax(h4, axis=1), tf.int32,
# # #                            name='labels')
# # #     }
    
# # #     ## Visualize the graph with TensorBoard:

# # #     ## Loss Function and Optimization
# # #     cross_entropy_loss = tf.reduce_mean(
# # #         tf.nn.softmax_cross_entropy_with_logits(
# # #             logits=h4, labels=tf_y_onehot),
# # #         name='cross_entropy_loss')


# # #     if norm == 'L1':
# # #         L1_norm = tf.norm(h3_weights, ord='fro', axis=(0,1))

# # #         ## Optimizer:
# # #         optimizer = tf.train.AdamOptimizer(learning_rate)
# # #         optimizer = optimizer.minimize(cross_entropy_loss+lamb*L1_norm,
# # #                                     name='train_op')

# # #     elif norm == 'L2':
# # #         L2_norm = tf.norm(h3_weights, ord='euclidean', axis=(0,1))

# # #         ## Optimizer:
# # #         optimizer = tf.train.AdamOptimizer(learning_rate)
# # #         optimizer = optimizer.minimize(cross_entropy_loss+lamb*L2_norm,
# # #                                     name='train_op')



# # #     ## Computing the prediction accuracy
# # #     correct_predictions = tf.equal(
# # #         predictions['labels'], 
# # #         tf_y, name='correct_preds')

# # #     accuracy = tf.reduce_mean(
# # #         tf.cast(correct_predictions, tf.float32),
# # #         name='accuracy')

# # # def save(saver, sess, epoch, path='./model/'):
# # #     if not os.path.isdir(path):
# # #         os.makedirs(path)
# # #     print('Saving model in %s' % path)
# # #     saver.save(sess, os.path.jo3in(path,'cnn-model.ckpt'),
# # #                global_step=epoch)

    
# # # def load(saver, sess, path, epoch):
# # #     print('Loading model from %s' % path)
# # #     saver.restore(sess, os.path.join(
# # #             path, 'cnn-model.ckpt-%d' % epoch))

    
# # # def train(sess, training_set, batch_size, validation_set=None,
# # #           initialize=True, epochs=20, shuffle=True,
# # #           dropout=0.5, random_seed=None):

# # #     X_data = np.array(training_set[0])
# # #     y_data = np.array(training_set[1])
# # #     training_loss = []

# # #     ## initialize variables
# # #     if initialize:
# # #         sess.run(tf.global_variables_initializer())
        
# # #     np.random.seed(random_seed) # for shuflling in batch_generator
# # #     for epoch in range(1, epochs+1):
# # #         batch_gen = batch_generator(
# # #                         X_data, y_data, batch_size,
# # #                         shuffle=shuffle)
# # #         avg_loss = 0.0
# # #         for i,(batch_x,batch_y) in enumerate(batch_gen):
# # #             feed = {'tf_x:0': batch_x, 
# # #                     'tf_y:0': batch_y, 
# # #                     'fc_keep_prob:0': dropout}
# # #             loss, _ = sess.run(
# # #                     ['cross_entropy_loss:0', 'train_op'],
# # #                     feed_dict=feed)
# # #             avg_loss += loss

# # #         training_loss.append(avg_loss / (i+1))
# # #         print('Epoch %02d Training Avg. Loss: %7.3f' % (epoch, avg_loss))
# # #         if validation_set is not None:
# # #             feed = {'tf_x:0': validation_set[0],
# # #                     'tf_y:0': validation_set[1],
# # #                     'fc_keep_prob:0':1.0} 
# # #             valid_acc = sess.run('accuracy:0', feed_dict=feed)
# # #             print(' Validation Acc: %7.3f' % valid_acc)
# # #         else:
# # #             print()

            
# # # def predict(sess, X_test, return_proba=False):
# # #     feed = {'tf_x:0': X_test, 
# # #             'fc_keep_prob:0': 1.0}
# # #     if return_proba:
# # #         return sess.run('probabilities:0', feed_dict=feed)
# # #     else:
# # #         return sess.run('labels:0', feed_dict=feed)


# # # keep_prob_param = [0.1, 0.5, 1.0]
# # # norm = ['L1', 'L2']
# # # lamb = [0, 0.1, 0.001]
# # # mini_batch = [1, 4, 16, 64, 128, 1024, 55000]



# # # if (sys.version_info > (3, 0)):
# # #     writemode = 'wb'
# # # else:
# # #     writemode = 'w'

# # # zipped_mnist = [f for f in os.listdir('./')
# # #                 if f.endswith('ubyte.gz')]
# # # for z in zipped_mnist:
# # #     with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
# # #         outfile.write(decompressed.read())

# # # X_data, y_data = load_mnist('./', kind='train')
# # # print('Rows: %d,  Columns: %d' % (X_data.shape[0], X_data.shape[1]))
# # # X_test, y_test = load_mnist('./', kind='t10k')
# # # print('Rows: %d,  Columns: %d' % (X_test.shape[0], X_test.shape[1]))

# # # X_train, y_train = X_data[:55000,:], y_data[:55000]
# # # X_valid, y_valid = X_data[55000:,:], y_data[55000:]

# # # print('Training:   ', X_train.shape, y_train.shape)
# # # print('Validation: ', X_valid.shape, y_valid.shape)
# # # print('Test Set:   ', X_test.shape, y_test.shape)


# # # mean_vals = np.mean(X_train, axis=0)
# # # std_val = np.std(X_train)

# # # X_train_centered = (X_train - mean_vals)/std_val
# # # X_valid_centered = X_valid - mean_vals
# # # X_test_centered = (X_test - mean_vals)/std_val

# # # print_('sys.getsizeof(X_data)')
# # # print_('sys.getsizeof(y_data)')
# # # print_('sys.getsizeof(X_train)')
# # # print_('sys.getsizeof(X_valid)')
# # # print_('sys.getsizeof(X_test)')
# # # print_('sys.getsizeof(X_train_centered)')
# # # print_('sys.getsizeof(X_valid_centered)')
# # # print_('sys.getsizeof(X_test_centered)')

# # # del X_data, y_data, X_train, X_valid, X_test



# # # # for i in keep_prob_param:
# # # #     for j in norm:
# # # #         for k in lamb:
# # # #             for l in mini_batch:
# # # i = keep_prob_param[0]
# # # j = norm[0]
# # # k = lamb[0]
# # # l = mini_batch[0]

# # # g = tf.Graph()
# # # with g.as_default():
# # #     tf.set_random_seed(random_seed)
    
# # #     build_cnn(j, k)
    
# # #     saver = tf.train.Saver()
# # # with tf.Session(graph=g) as sess:
# # #     train(sess, training_set=(X_train_centered, y_train), batch_size=l,
# # #         validation_set=(X_valid_centered, y_valid), initialize=True, random_seed=123, dropout=i, epochs=2)
# # #     save(saver, sess, epoch=2)

# # # del g


# # # g2 = tf.Graph()
# # # with g2.as_default():
# # #     tf.set_random_seed(random_seed)
    
# # #     build_cnn(norm[0], lamb[0])

# # #     ## saver:
# # #     saver = tf.train.Saver()

# # # ## create a new session 
# # # ## and restore the model
# # # with tf.Session(graph=g2) as sess:
# # #     load(saver, sess, epoch=20, path='./model/')
    
# # #     preds = predict(sess, X_test_centered, return_proba=False)

# # #     print('Test Accuracy: %.3f%%' % (100*np.sum(preds == y_test)/len(y_test)))    