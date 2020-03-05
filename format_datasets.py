from keras.datasets import cifar10
import matplotlib.pyplot as plt
import keras.utils as utils
import numpy as np

# if you want, you can define train_images,train_labels,test_images, test_labels
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
first_image = x_train[0]

# block 1
# show images
# print(first_image[0])
# plt.imshow(first_image)
# plt.show()



labels_array = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# block 2
# first_label_index = y_train[0][0]
# #printing out an index into an array of categories
# print(y_train[0])
#
# #printing out an labelled array of categories
# print(labels_array[first_label_index])


# block 3
#y_train = utils.to_categorical(y_train)
#max_index = np.argmax(y_train[0])
# find a max value of array
#print(labels_array[max_index])


# # not important!!!
# # block 4
# # one hot encoding
# def reshape_image(input_image_arrays):
#     output_array = []
#     for image_array in input_image_arrays:
#         output_array.append(image_array.reshape(-1))
#     return np.asarray(output_array)
#
#
# y_train = utils.to_categorical(y_train)
# y_test = utils.to_categorical(y_test)
#
# x_train = x_train.astype('float32')
# x_train = x_train/255.0
# y_train = y_train.astype('float32')
# y_train = y_train/255.0
#
# print(y_train[0])