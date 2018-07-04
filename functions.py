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



def build_cnn(norm='L1', lamb=0):
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


def unzip(end='ubyte.gz'):
    if (sys.version_info > (3, 0)):
        writemode = 'wb'
    else:
        writemode = 'w'

    zipped_mnist = [f for f in os.listdir('./')
                    if f.endswith(end)]
    for z in zipped_mnist:
        with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
            outfile.write(decompressed.read())


def LoadTrainData(path, kind):

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        LabelData = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        ImageData = np.fromfile(imgpath, dtype=np.uint8).reshape(len(LabelData), 784)

    return ImageData, LabelData

def activated_nodes(input_node, weights, FormalRedValue, scope=0.999):

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
        print('All elements in input_node are 0')
        sys.exit(1) 
    for index, i in enumerate(list(SortedMulMember)):
        if i > 0:
            sum_2 += i
            if sum_2 / sum_1 > scope:
                check_point = index
                break

    nodes = sorted_index[np.arange(check_point)]
    RedValue = SortedMulMember[np.arange(check_point)]*FormalRedValue/sum_1

    return nodes, RedValue




def unpooling(RedPixelsBeforeUnpool, ImageAfterUnpool):
    '''input : RedPixelsBeforeUnpool(3919, 4), ImageAfterUnpool(1, 24, 24, 32), '''
    print('<unpooling>')

    RedPixelsAfterUnpool = np.array([[]], dtype="i")

    for i in range(ImageAfterUnpool.shape[3]):
        conv1_tmp = np.array(ImageAfterUnpool[0,:,:,i])
        
        for index, channel in enumerate(RedPixelsBeforeUnpool[:,2]):
            if channel == i:
                column = int(RedPixelsBeforeUnpool[index,0])
                row = int(RedPixelsBeforeUnpool[index,1])

                pixel_4x4 = [float(conv1_tmp[2*column, 2*row]), \
                            float(conv1_tmp[2*column+1, 2*row]), \
                            float(conv1_tmp[2*column, 2*row+1]), \
                            float(conv1_tmp[2*column+1, 2*row+1])]
                maximum_value = max(pixel_4x4)
                maximum_index = pixel_4x4.index(maximum_value)

                if maximum_index == 0:
                    column = 2*column
                    row = 2*row
                elif maximum_index == 1:
                    column = 2*column+1
                    row = 2*row
                elif maximum_index == 2:
                    column = 2*column
                    row = 2*row+1
                elif maximum_index == 3:
                    column = 2*column+1
                    row = 2*row+1

                RedPixelsAfterUnpool = \
                            np.c_[RedPixelsAfterUnpool, [[column, row, channel, RedPixelsBeforeUnpool[index,3]]]]
                
                
    RedPixelsAfterUnpool = np.resize(RedPixelsAfterUnpool,(int(RedPixelsAfterUnpool.shape[1]/4), 4))

    '''output : RedPixelsAfterUnpool(All red pixel's coord and channel)'''       
    return RedPixelsAfterUnpool




def CollectRedValues(RedPixelsBeforeCollect, CorrespondingLayer):
    '''input : RedPixelsBeforeCollect(1, ~, 4)(column, row, channel, redvalue), CorrespondingLayer(1, 28, 28)'''
    
    if RedPixelsBeforeCollect.shape[2] == 3:
        TempArray = np.zeros((CorrespondingLayer.shape))
        for i in RedPixelsBeforeCollect[0,:,:]:
            TempArray[0,int(i[0]),int(i[1])] += i[2]

        RedPixelsAfterCollect = np.array([[]])
        for i in range(CorrespondingLayer.shape[1]):
            for j in range(CorrespondingLayer.shape[2]):
                if TempArray[0,i,j] != 0:
                    RedPixelsAfterCollect = np.c_[RedPixelsAfterCollect,np.array([[i,j,TempArray[0,i,j]]])]


    elif RedPixelsBeforeCollect.shape[2] == 4:
        TempArray = np.zeros((CorrespondingLayer.shape))
        for i in RedPixelsBeforeCollect[0,:,:]:
            TempArray[0,int(i[0]),int(i[1]),int(i[2])] += i[3]

        RedPixelsAfterCollect = np.array([[]])
        for i in range(CorrespondingLayer.shape[1]):
            for j in range(CorrespondingLayer.shape[2]):
                for k in range(CorrespondingLayer.shape[3]):
                    if TempArray[0,i,j,k] != 0:
                        RedPixelsAfterCollect = np.c_[RedPixelsAfterCollect, np.array([[i,j,k,TempArray[0,i,j,k]]])]

    RedPixelsAfterCollect = np.resize(RedPixelsAfterCollect, \
                                (int(RedPixelsAfterCollect.shape[1]/RedPixelsBeforeCollect.shape[2]), \
                                                                    RedPixelsBeforeCollect.shape[2]))

    '''output : RedPixelsAfterCollect(far_less_amount_of (column,row,channel,redvalue))'''
    return RedPixelsAfterCollect





def ShowImage(ImageToBeRed, PixelsToBeRed, TrueNum, threshold, correct, probabilities, ShowNotSave, RowNumber=8, ColNumber=8):
    '''input : image(1, 784) , RowNumber, ColNumber,
    PixelsToBeRed(618, 3)(All red pixel's coord and channel | column, row, RedValue)'''

    '''Normalizing Red Value(from 0 to 1)'''
    PixelsToBeRed_Nor = np.empty_like(PixelsToBeRed)
    np.copyto(PixelsToBeRed_Nor, PixelsToBeRed)
    RedValueColumn = PixelsToBeRed.shape[1]-1
    PixelsToBeRed_Nor[:,RedValueColumn] = (PixelsToBeRed_Nor[:,RedValueColumn] - PixelsToBeRed_Nor[:,RedValueColumn].min())/ \
                                    (PixelsToBeRed_Nor[:,RedValueColumn].max() - PixelsToBeRed_Nor[:,RedValueColumn].min())


    '''Remove under certain value'''
    remove_list = np.array([[]])
    for index, i in enumerate(PixelsToBeRed_Nor[:,RedValueColumn]):
        if i <= threshold: #@@@@
            remove_list = np.c_[remove_list, [index]]
    PixelsToBeRed_Nor = np.delete(PixelsToBeRed_Nor, remove_list, axis=0)
    PixelsToBeRedDeleted = np.delete(PixelsToBeRed, remove_list, axis=0)


    '''Debugging'''
    # np.set_printoptions(threshold=np.inf)
    # print_('PixelsToBeRed')
    # print_('np.sum(PixelsToBeRedDeleted[:,2])')
    # print_('PixelsToBeRedDeleted[:,2]')
    # print_('PixelsToBeRed.shape')
    # print_('np.sum(PixelsToBeRed[:,2])')
    # np.set_printoptions(threshold=1000)



    '''image plot''' 
    if len(ImageToBeRed.shape) == 3:
        ImageToBeRed_ = ImageToBeRed[0,:,:]
        ImageToBeRed_norm = (ImageToBeRed_ - ImageToBeRed_.min()) / (ImageToBeRed_.max() - ImageToBeRed_.min())
        # ImageToBeRed_norm = np.resize(ImageToBeRed_norm, (28,28))
        ImageToBePlot = np.stack([ImageToBeRed_norm, ImageToBeRed_norm, ImageToBeRed_norm], axis=2)

        for OnePixelToBeRed in PixelsToBeRed_Nor:
            # print_('OnePixelToBeRed[2]')
            ImageToBePlot[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),0] = OnePixelToBeRed[2]
            ImageToBePlot[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),1] = 0
            ImageToBePlot[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),2] = 0
        
        plt.imshow(ImageToBePlot)


    elif len(ImageToBeRed.shape) == 4:
        fig, ax_ImageToBeRed = plt.subplots(nrows=RowNumber, ncols=ColNumber, figsize=(5, 5))
        AllChannels = ImageToBeRed[0,:,:,:]
        for i in range(ImageToBeRed.shape[3]):
            OneChannel =  AllChannels[:,:,i]    
            ImageToBePlot = np.stack([OneChannel, OneChannel, OneChannel], axis=2)
            for index, channel in enumerate(np.array(PixelsToBeRed_Nor[:,2], dtype="i")):
                if channel == i:
                    OnePixelToBeRed = PixelsToBeRed_Nor[index]
                    ImageToBePlot[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),0] = OnePixelToBeRed[3] +0.5
                    ImageToBePlot[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),1] = 0
                    ImageToBePlot[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),2] = 0
            
            ax_ImageToBeRed[int(i/8), int(i%8)].imshow(ImageToBePlot)

    if ShowNotSave == True:
        plt.show()
    else:
        threshold = "%0.2f" %threshold
        probabilities = "%0.2f" %probabilities
        filename = 'No.'+str(TrueNum)+'_'+threshold+'_'+str(correct[0])+'_'+probabilities+' .png'
        print(filename, ' saved')
        plt.savefig(filename)

    '''output : showing images'''


def Deconvolution(RedPixelsBeforeDeconv, kernel, ImageAfterDeconv, scope=0.999):
    # '''input : image(1, 784), kernel(5, 5, 1, 32), RedPixelsBeforeDeconv(3919, 4)(column, row, channel, redvalue)'''
    print('<Deconvolution>')
    RedPixelsAfterDeconv = np.array([[]])


    ImageAfterDeconv_resize = ImageAfterDeconv[0,:,:]

    for i in range(kernel.shape[3]): 
        for index, channel in enumerate(np.array(RedPixelsBeforeDeconv[:,2], dtype="i")):
            if channel == i:
                StartPointOfCutting = RedPixelsBeforeDeconv[index,:]
                CuttedPart = ImageAfterDeconv_resize[int(StartPointOfCutting[0]):int(StartPointOfCutting[0])+4,\
                                                     int(StartPointOfCutting[1]):int(StartPointOfCutting[1])+4]
                kernel_ = kernel[:,:,:,i]
                
                length = kernel_.shape[0] * kernel_.shape[1] * kernel_.shape[2]
                CuttedPart1D = np.resize(CuttedPart, (1,length))
                kernel_1D = np.resize(kernel_, (1,length))
                
                RedPixelsPointsAfterDeconv1D, RedValueAfterDeconv = \
                    activated_nodes(CuttedPart1D, kernel_1D, StartPointOfCutting[3])

                if kernel_.shape[2] == 1:
                    for index, RedPixelPoint1D in enumerate(RedPixelsPointsAfterDeconv1D):
                        RedPixelPointcolumn = StartPointOfCutting[0]+RedPixelPoint1D%kernel_.shape[0]
                        RedPixelPointrow = StartPointOfCutting[1]+int(RedPixelPoint1D/kernel_.shape[0])
                        RedPixelsAfterDeconv = np.c_[RedPixelsAfterDeconv, \
                                        [[RedPixelPointcolumn,RedPixelPointrow,RedValueAfterDeconv[index]]]]
                else:
                    for index, RedPixelPoint1D in enumerate(RedPixelsPointsAfterDeconv1D):
                        RedPixelPointcolumn = StartPointOfCutting[0]+int(RedPixelPoint1D/(kernel_.shape[2]*kernel_.shape[1]))
                        RedPixelPointrow = StartPointOfCutting[1]+int(RedPixelPoint1D%(kernel_.shape[2]*kernel_.shape[1])/kernel_.shape[2])
                        RedPixelPointChannel = RedPixelPoint1D%(kernel_.shape[2]*kernel_.shape[1])%kernel_.shape[2]
                        RedPixelsAfterDeconv = np.c_[RedPixelsAfterDeconv, \
                                        [[RedPixelPointcolumn,RedPixelPointrow,RedPixelPointChannel,RedValueAfterDeconv[index]]]]

    RedPixelsAfterDeconv = np.resize(RedPixelsAfterDeconv, (1,\
                            int(RedPixelsAfterDeconv.shape[1]/(RedPixelsBeforeDeconv.shape[1]-(kernel_.shape[2]==1))),\
                                                            RedPixelsBeforeDeconv.shape[1]-(kernel_.shape[2]==1)))

    # '''output : showing images, RedPixelsAfterDeconv(All red pixel's coord and channel)'''
    return RedPixelsAfterDeconv
    



def Fcl2PoolingLayer(RedNodesInFcl, FclWeights, PoolingLayer, RedValue):
    # '''input : PoolingLayer(1, 4, 4, 64), RedNodesInFcl(210,), FclWeights(1024, 1024), RedValue'''
    print('<Fcl2PoolingLayer>')
    AllRedPixelsInPoolingLayer = np.array([[]])
    AllRedValueInPoolingLayer = np.array([[]])
    PoolingLayer_resize = np.resize(PoolingLayer[0,:,:,:], (1,1024))

    for num in range(RedNodesInFcl.shape[0]):
        RedPixelsPointsInPoolingLayer, RedValueInPoolingLayer = \
                activated_nodes(PoolingLayer_resize, FclWeights[:,RedNodesInFcl[num]], RedValue[num]) ####
        AllRedPixelsInPoolingLayer = np.c_[AllRedPixelsInPoolingLayer, [RedPixelsPointsInPoolingLayer]]
        AllRedValueInPoolingLayer = np.c_[AllRedValueInPoolingLayer, [RedValueInPoolingLayer]]

    columns = np.array(AllRedPixelsInPoolingLayer/(64*4), dtype="i")
    rows = np.array(AllRedPixelsInPoolingLayer%(64*4)/64, dtype="i")
    channels = np.array(AllRedPixelsInPoolingLayer%(64*4)%64, dtype="i")

    RedPixelsInPoolingLayer = np.stack([columns, rows, channels, AllRedValueInPoolingLayer], axis=2)
    
    # '''output : RedPixelsInPoolingLayer(All red pixels' coord and channel)'''
    return RedPixelsInPoolingLayer
    