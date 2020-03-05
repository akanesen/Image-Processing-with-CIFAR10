from keras.datasets import cifar10
import keras.utils as utils
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# One Hot Encoding for target values
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

model = load_model('Image_Classifier.h5')

results = model.evaluate(x=x_test, y=y_test)

test_image_data = np.asarray([x_test[5000]])
print(model.predict(x=test_image_data))

# print highest predict
prediction = model.predict(x=test_image_data)
max_index = np.argmax(prediction[0])
print("Prediction: ", labels[max_index])

plt.imshow(x_test[5000])
plt.show()

"""
# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	plt.subplot(211)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	plt.subplot(212)
	plt.title('Classification Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	plt.savefig(filename + '_plot.png')
	plt.close()

summarize_diagnostics(history)
"""