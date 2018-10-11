
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, InputLayer
from keras.layers import Conv2D, MaxPooling2D

# get dataset
from keras.datasets import mnist

import matplotlib.pyplot as plt

import itertools

def generate_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train,(60000,28,28,1))
    x_test = np.reshape(x_test,(10000,28,28,1))
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test


# class ModelConfig():
    
#     def __init__(self, config):
#         self.filter_1_size = config[0]
#         self.num_of_filter_in_layer_1 = config[1]
#         self.filter_2_size = config[2]
#         self.num_of_filter_in_layer_2 = config[3]

# def model_config_generation():
#     filter_1_size = [(3, 3), (4, 4), (5, 5)]
#     num_of_filter_in_layer_1 = [2, 4, 8, 16, 32]
#     filter_2_size = [(3, 3), (4, 4), (5, 5)]
#     num_of_filter_in_layer_2 = [2, 4, 8, 16, 32]

#     # filter_1_size = [(3, 3)]
#     # num_of_filter_in_layer_1 = [16]
#     # filter_2_size = [(3, 3), (5, 5)]
#     # num_of_filter_in_layer_2 = [16]

#     config_setting = [filter_1_size, num_of_filter_in_layer_1,
#         filter_2_size, num_of_filter_in_layer_2]
    
#     config_set = list(itertools.product(*config_setting))
#     config_set = list(map(ModelConfig, config_set))

#     return config_set

def model_generator(hidden_num, activation='relu'):
    print('model_' + str(hidden_num) + ' build')
    model = Sequential()
    model.add(InputLayer((28,28,1)))
    model.add(Flatten())
    model.add(Dense(hidden_num, activation = activation))
    model.add(Dense(10, activation = 'softmax', name = 'end'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    return model
    # print('model build')
    # input = Input(shape=(28,28,1,), name='input')
    # layer = Conv2D(model_config.num_of_filter_in_layer_1, model_config.filter_1_size, activation=activation, name='layer1')(input)
    # layer = MaxPooling2D(pool_size=(2, 2))(layer)
    # layer = Conv2D(model_config.num_of_filter_in_layer_2, model_config.filter_2_size, activation=activation, name='layer2')(input)
    # layer = MaxPooling2D(pool_size=(2, 2))(layer)
    # layer = Flatten()(layer)
    # end = Dense(10, activation='softmax', name='end')(layer)     
    # model = Model(input=input, output=end)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # return model

    
def train_and_get_loss_acc_and_param_size(model, x_train, y_train, x_test, y_test, epochs=250, batch_size=64):
    res = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))

    return model.count_params(), res.history['acc'], res.history['loss'], res.history['val_acc'], res.history['val_loss']

def plot_result(x, y1, y2, title, label1, label2, x_tile, y_title, fig_name):
    colors = (0,0,0)
    area = np.pi*3
    
    # Plot
    plt.scatter(x, y1, s=area, c='green', alpha=0.5, label=label1)
    plt.scatter(x, y2, s=area, c='red', alpha=0.5, label=label1)
    plt.title(title)
    plt.xlabel(x_tile)
    plt.ylabel(y_title)
    plt.legend()
    plt.savefig(fig_name+'.png')
    plt.gcf().clear()

def main():
    # config_set = model_config_generation()

    x_train, y_train, x_test, y_test = generate_data()

    model_param_size = []
    model_train_acc = []
    model_train_loss = []
    model_test_acc = []
    model_test_loss = []

    for i in range(1, 5):
        model = model_generator(i)
        param_size, acc, loss, val_acc, val_loss = train_and_get_loss_acc_and_param_size(
            model, x_train, y_train, x_test, y_test, epochs=5)
        model_param_size.append(param_size)
        model_train_acc.append(acc)
        model_train_loss.append(loss)
        model_test_acc.append(val_acc)
        model_test_loss.append(val_loss)

    plot_result(model_param_size, model_train_loss, model_test_loss,
        'MNIST Model Loss','train_loss', 'test_loss', 'number of parameters', 'loss', '3b_loss')
    plot_result(model_param_size, model_train_acc, model_test_acc,
        'MNIST Model Acc','train_acc', 'test_acc', 'number of parameters', 'acc', '3b_acc')




if __name__ == '__main__':
    main()