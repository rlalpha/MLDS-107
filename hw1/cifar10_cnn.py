
# coding: utf-8

# In[1]:


from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer
from hw1_2 import model_generator_1, model_generator_2


def model_generator_3(input_shape, no_classes):
	print('model_3 build')
	model = Sequential()
	model.add(InputLayer(input_shape))
	model.add(Flatten())

	model.add(Dense(128))
	model.add(Activation('relu'))

	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(no_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam', metrics=['accuracy'])

	print(model.summary())
	return model

def model_generator_4(input_shape, no_classes):
	print('model_4 build')
	model = Sequential()
	model.add(InputLayer(input_shape))
	model.add(Flatten())

	# Dense Layer * 12
	for i in range(8):
		model.add(Dense(128))
		model.add(Activation('relu'))

	model.add(Dense(no_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam', metrics=['accuracy'])

	print(model.summary())
	return model

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def train(x_train, y_train, x_test, y_test, epochs, datasetName):

	input_shape, no_classes = x_train.shape[1:], y_train.shape[1]

	model_1 = model_generator_1(input_shape, no_classes)
	model_2 = model_generator_2(input_shape, no_classes)
	model_3 = model_generator_3(input_shape, no_classes)
	model_4 = model_generator_4(input_shape, no_classes)

	history_1 = model_1.fit(x_train, y_train, batch_size=1000,
							epochs=epochs, verbose=1, validation_data=(x_test, y_test))
	history_2 = model_2.fit(x_train, y_train, batch_size=1000,
							epochs=epochs, verbose=1, validation_data=(x_test, y_test))
	history_3 = model_3.fit(x_train, y_train, batch_size=1000,
							epochs=epochs, verbose=1, validation_data=(x_test, y_test))
	history_4 = model_4.fit(x_train, y_train, batch_size=1000,
							epochs=epochs, verbose=1, validation_data=(x_test, y_test))

	# val loss of all models
	plt.plot(history_1.history['val_loss'],
			 color='blueviolet', label='cnn-model-1-deep')
	plt.plot(history_2.history['val_loss'],
			 color='blue', label='cnn-model-2-shallow')
	plt.plot(history_3.history['val_loss'],
			 color='pink', label='dnn-model-1-shallow')
	plt.plot(history_4.history['val_loss'],
			 color='red', label='dnn-model-2-deep')
	plt.title("%s Val Loss Plot"%datasetName)
	plt.legend()
	plt.savefig('cnn_%s_val_loss_%i_comparison_of_deep_and_shallow.png' %
				(datasetName, train.counter))
	plt.gcf().clear()

	# val acc of all models
	plt.plot(history_1.history['val_acc'],
			 color='blueviolet', label='cnn-model-1-deep')
	plt.plot(history_2.history['val_acc'],
			 color='blue', label='cnn-model-2-shallow')
	plt.plot(history_3.history['val_acc'],
			 color='pink', label='dnn-model-1-shallow')
	plt.plot(history_4.history['val_acc'],
			 color='red', label='dnn-model-2-deep')
	plt.title("%s Val Acc Plot"%datasetName)
	plt.legend()
	plt.savefig('cnn_%s_val_acc_%i_comparison_of_deep_and_shallow.png' % (datasetName, train.counter))
	plt.gcf().clear()

	# train loss of all models
	plt.plot(history_1.history['loss'],
			 color='blueviolet', label='cnn-model-1-deep')
	plt.plot(history_2.history['loss'],
			 color='blue', label='cnn-model-2-shallow')
	plt.plot(history_3.history['loss'],
			 color='pink', label='dnn-model-1-shallow')
	plt.plot(history_4.history['loss'],
			 color='red', label='dnn-model-2-deep')
	plt.title("%s Loss Plot"%datasetName)
	plt.legend()
	plt.savefig('cnn_%s_val_loss_%i_comparison_of_deep_and_shallow.png' %
				(datasetName, train.counter))
	plt.gcf().clear()

	# train acc of all models
	plt.plot(history_1.history['acc'],
			 color='blueviolet', label='cnn-model-1-deep')
	plt.plot(history_2.history['acc'],
			 color='blue', label='cnn-model-2-shallow')
	plt.plot(history_3.history['acc'],
			 color='pink', label='dnn-model-1-shallow')
	plt.plot(history_4.history['acc'],
			 color='red', label='dnn-model-2-deep')
	plt.title("%s Acc Plot"%datasetName)
	plt.legend()
	plt.savefig('cnn_%s_acc_%i_comparison_of_deep_and_shallow.png' %
				(datasetName, train.counter))
	plt.gcf().clear()

	train.counter += 1

if __name__ == '__main__':

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)

	print(x_train.shape)
	print(y_train.shape)
	print(x_test.shape)
	print(y_test.shape)
	train.counter = 0
	train(x_train, y_train, x_test, y_test, 20, 'cifar-10')
