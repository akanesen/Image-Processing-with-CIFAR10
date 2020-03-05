from keras.datasets import cifar10
import keras.utils as utils
from keras.optimizers import SGD

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#normalize to range 0-1
(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') /255.0
x_test = x_test.astype('float32') /255.0

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)


# Create the model as a sequential type so that we can add layers in order
model = Sequential()

# Add the first convolution to output a feature map
# filters: output 32 features
# kernel_size: 3x3 kernel or filter matrix used to calculate output features
# input_shape: each image is 32x32x3
# activation: relu activation for each of the operations as it produces the best results
# padding: 'same' adds padding to the input image to make sure that the output feature map is the same size as the input
# kernel_constraint: maxnorm normalizes the values in the kernel to make sure that the max value is 3
model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 input_shape=(32,32,3),
                 activation='relu',
                 padding = 'same',
                kernel_constraint = maxnorm(3)))

# Add the max pool layer to decrease the image size from 32x32 to 16x16
# pool_size: finds the max value in each 2x2 section of the input
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten layer converts a matrix into a 1 dimensional array
model.add(Flatten())

# First dense layer to create the actual prediction network
# units: 512 neurons at this layer, increase for greater accuracy, decrease for faster train speed
# activation: relu because it works so well
# kernel_constraint: see above
model.add(Dense(units=512, activation='relu',
                kernel_constraint=maxnorm(3)))

# Dropout layer to ignore some neurons during training which improves model reliability
# rate: 0.5 means half neurons dropped
model.add(Dropout(rate=0.5))

# Final dense layer used to produce output for each of the 10 categories
# units: 10 categories so 10 output units
# activation: softmax because we are calculating probabilities for each of the 10 categories (not as clear as 0 or 1)
model.add(Dense(units=10, activation='softmax'))

# compile model
model.compile(optimizer=SGD(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fit model
model.fit(x=x_train, y=y_train, epochs=30, batch_size=32)

# save model
model.save(filepath='Image_Classifier.h5')
