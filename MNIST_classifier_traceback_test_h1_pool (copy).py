import MNIST_classifier as third
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import struct
import gzip
import sys
import os


random_seed = 123
np.random.seed(random_seed)



def print_(name):
    print(name, ' : \n', eval(name))




third.unzip('ubyte.gz')

X_data = np.array([[]])
y_data = np.array([[]])
X_data, y_data = third.LoadTrainData('./', 'train') # or 'train'
X_test, y_test = third.LoadTrainData('./', 't10k') # or 'train'

X_train, y_train = X_data[:55000,:], y_data[:55000]
X_valid, y_valid = X_data[55000:,:], y_data[55000:]

number = 30
X_test, y_test = X_test[number:number+1,:], y_test[number:number+1]

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_valid_centered = X_valid - mean_vals
X_test_centered = (X_test - mean_vals)/std_val

del X_data, y_data, X_train, X_valid, X_test



g2 = tf.Graph()
with g2.as_default():
    tf.set_random_seed(random_seed)
    
    third.build_cnn()

    saver = tf.train.Saver()


with tf.Session(graph=g2) as sess:

    third.load(saver, sess, epoch=2, path='./model/')

    feed = {'tf_x:0': X_test_centered, 
            'fc_keep_prob:0': 1.0}

    image, Conv1Kernel, Conv1Bias, conv_1_result, result_of_h1_pool, \
    Conv2Kernel, Conv2Bias, conv_2_result, fc_3_w, h2_pool, fc_3_b, fc_4_w, \
    fc_3_result, fc_4_b, fc_4_result, preds = sess.run(['tf_x:0', 'conv_1/_weights:0', \
    'conv_1/_biases:0', 'conv_1/activation:0', 'h1_pool:0', 'conv_2/_weights:0', 'conv_2/_biases:0',\
        'conv_2/activation:0', 'fc_3/_weights:0', 'h2_pool:0', 'fc_3/_biases:0', 'fc_4/_weights:0',\
    'fc_3/activation:0', 'fc_4/_biases:0', 'fc_4/net_pre-activation:0','labels:0'], feed_dict=feed)

    image = np.resize(image, (1,28,28))
    # img = np.resize(X_test_img * 1/255, (28,28))
    # img = np.stack([img, img, img], axis=2)

                




'''=============================fc3 <- fc4================================'''
'''input : fc_4_result(1, 10)'''
# print_('fc_3_result.shape') # (1, 1024)
# print_('np.array([fc_4_w[:,7]]).shape') # (1, 1024)
# print_('fc_4_b[7].shape') # Just a constant 
PixelsToBeRedFc3, Fc3RedValue = third.activated_nodes(fc_3_result, np.array([fc_4_w[:,0]]), fc_4_b[7], 1) #@@@@
'''======================================================================='''




'''============================h2_pool <- fc3============================'''
'''input : h2_pool(1, 4, 4, 64), PixelsToBeRedFc3(210,), fc_3_w(1024, 1024), fc_3_b(1024,)'''
print('============================h2_pool <- fc3============================')
AllRedPixelsPointsFromH2pool = np.array([[]])
AllH2poolRedValue = np.array([[]])
h2_pool_resize = np.resize(h2_pool[0,:,:,:], (1,1024))

for num in range(PixelsToBeRedFc3.shape[0]):
    RedPixelsPointsFromH2pool, H2poolRedValue = \
        third.activated_nodes(h2_pool_resize, fc_3_w[:,PixelsToBeRedFc3[num]], fc_3_b[num], Fc3RedValue[num]) ####
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


PixelsToBeRedH2pool = third.CollectRedValues(PixelsToBeRedH2pool, h2_pool)


# # # print_('PixelsToBeRedH2pool.shape') # (1024, 4)   
# # # print_('np.sum(PixelsToBeRedH2pool[:,3])') # 0.9913220198592825




PixelsToBeRedConv2 = third.unpooling(PixelsToBeRedH2pool, conv_2_result)


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
                third.activated_nodes(Pixel1x800ValuefromH1pool, Kernel1x800Value, Conv2Bias[channel], StartPointOfCutting5x5[3])
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



PixelsToBeRedH1pool = third.CollectRedValues(PixelsToBeRedH1pool, result_of_h1_pool)




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




PixelsToBeRedConv1 = third.unpooling(PixelsToBeRedH1pool, conv_1_result)



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


Image_resize = image[0,:,:]
print_('image')

for i in range(Conv1Kernel.shape[2]): 
    for index, channel in enumerate(np.array(PixelsToBeRedConv1[:,2], dtype="i")):
        if channel == i:
            StartPointOfCutting5x5 = PixelsToBeRedConv1[index,:]
            Pixel5x5ValuefromImage = Image_resize[int(StartPointOfCutting5x5[0]):int(StartPointOfCutting5x5[0])+4,\
                                                int(StartPointOfCutting5x5[1]):int(StartPointOfCutting5x5[1])+4]
            Kernel5x5Value = Conv1Kernel[:,:,i]
            
            length = Kernel5x5Value.shape[0] * Kernel5x5Value.shape[1]
            Pixel1x25ValuefromImage = np.resize(Pixel5x5ValuefromImage, (1,length))
            Kernel1x25Value = np.resize(Kernel5x5Value, (1,length))
            
            RedPixelsPointsFromImage1D, ImageRedValue = \
                third.activated_nodes(Pixel1x25ValuefromImage, Kernel1x25Value, Conv1Bias[channel], StartPointOfCutting5x5[3])
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




PixelsToBeRedImage = third.CollectRedValues(PixelsToBeRedImage, image)



print_('PixelsToBeRedImage.shape') # (618, 3)
print_('np.sum(PixelsToBeRedImage[:,2])') # 0.07562872974755308





#     # print_('PixelsToBeRedImage')
# '''===================================Show Red Pixels===================================='''
# print('===================================Show Red Pixels====================================')
# '''input : image(1, 784) , RowNumber, ColNumber, **threshold**<if i <= 0.1:>
# PixelsToBeRedImage(618, 3)(All red pixel's coord and channel | column, row, RedValue)'''

# # plt.figure(0)
# # fig.tight_layout()

# '''Normalizing (from 0 to 1)'''
# print_('PixelsToBeRedImage[:,2].max()')
# PixelsToBeRedImage_Nor = np.empty_like(PixelsToBeRedImage)
# np.copyto(PixelsToBeRedImage_Nor, PixelsToBeRedImage)
# PixelsToBeRedImage_Nor[:,2] = (PixelsToBeRedImage_Nor[:,2] - PixelsToBeRedImage_Nor[:,2].min())/ \
#                                 (PixelsToBeRedImage_Nor[:,2].max() - PixelsToBeRedImage_Nor[:,2].min())
# print_('PixelsToBeRedImage[:,2].max()')


# '''Remove under certain value'''
# remove_list = np.array([[]])
# for index, i in enumerate(PixelsToBeRedImage_Nor[:,2]):
#     if i <= 0.6: #@@@@
#         remove_list = np.c_[remove_list, [index]]
# PixelsToBeRedImage_Nor = np.delete(PixelsToBeRedImage_Nor, remove_list, axis=0)
# PixelsToBeRedImageDeleted = np.delete(PixelsToBeRedImage, remove_list, axis=0)


# '''Debugging'''
# np.set_printoptions(threshold=np.inf)
# # print_('PixelsToBeRedImage')
# print_('np.sum(PixelsToBeRedImageDeleted[:,2])')
# print_('PixelsToBeRedImageDeleted[:,2]')
# print_('PixelsToBeRedImage.shape')
# print_('np.sum(PixelsToBeRedImage[:,2])')
# np.set_printoptions(threshold=1000)


# '''image plot'''
# # print_('image[0]')
# image_norm = (image - image.min()) / (image.max() - image.min())
# image_norm = np.resize(image_norm, (28,28))
# FinalImage = np.stack([image_norm, image_norm, image_norm], axis=2)

# for OnePixelToBeRed in PixelsToBeRedImage_Nor:
#     # print_('OnePixelToBeRed[2]')
#     FinalImage[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),0] = OnePixelToBeRed[2]
#     FinalImage[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),1] = 0
#     FinalImage[OnePixelToBeRed[0].astype(int), OnePixelToBeRed[1].astype(int),2] = 0

# # plt.imshow(FinalImage)
# # plt.show()

# # np.set_printoptions(threshold=np.inf)
# # print_('FinalImage.shape')
# # # print_('FinalImage')
# # np.set_printoptions(threshold=1000)

# plt.imshow(FinalImage)
# # plt.show()
# filename = 'Final_image_No1_0.20 .png'
# print(filename, ' saved')
# plt.savefig(filename)     
# # # del h2_pool  
# # '''output : showing images'''
# # '''====================================================================================='''



third.ShowImage(image, PixelsToBeRedImage,'Final_image_No1_0.20 .png', threshold=0.6)