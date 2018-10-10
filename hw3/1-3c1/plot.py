# basic
import keras.models
import numpy as np
# data set
from keras.datasets import mnist
from keras.utils import np_utils
# pic
from matplotlib import pyplot as plt


model = keras.models.load_model('1000_model.h5')
model_1 = keras.models.load_model('1000_model.h5')
model_2 = keras.models.load_model('2000_model.h5')


def generate_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('reshape')
    x_train = np.reshape(x_train,(60000,28,28,1))
    x_test = np.reshape(x_test,(10000,28,28,1))
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (x_train, y_train, x_test, y_test)
  
  
def evaluate(alpha, x, y):
    for layer, layer1, layer2 in zip(model.layers, model_1.layers, model_2.layers):
        new_weight = multi_add(layer1.get_weights(), layer2.get_weights(), alpha)
        layer.set_weights(new_weight) 
    loss, acc = model.evaluate(x[:5000], y[:5000], batch_size=5000)
    return loss, acc  

    
def main():
    print('load_data')
    alpha = np.linspace(-1, 2, 1000)
    x_train, y_train, x_test, y_test = generate_data()
    loss_train = []
    accuracy_train = []
    loss_test = []
    accuracy_test = []
    print('evaluate')
    for i in range(len(alpha)):
        print('alpha = ', alpha[i])
        l, a = evaluate(alpha[i], x_train, y_train)
        loss_train.append(l)
        accuracy_train.append(a) 
        l, a = evaluate(alpha[i], x_test, y_test)
        loss_test.append(l)
        accuracy_test.append(a)      
#     loss_train = np.array(loss_train)
#     np.save('loss_train.npy', loss_train)
#     loss_test = np.array(loss_test)
#     np.save('loss_test.npy', loss_test)
#     accuracy_train = np.array(accuracy_train)
#     np.save('accuracy_train.npy', accuracy_train)
#     accuracy_test = np.array(accuracy_test)
#     np.save('accuracy_test.npy', accuracy_test)
#     print('load_data')
#     loss_train = np.load('loss_train.npy')
#     loss_test = np.load('loss_test.npy')
#     accuracy_train = np.load('accuracy_train.npy')
#     accuracy_test = np.load('accuracy_test.npy')
    print('plot')       
    plt.xlabel('alpha')
    plt.ylabel('loss')
    plt.plot(alpha, loss_train, '-', color='r', label='loss_train')    # 點和線
    plt.plot(alpha, loss_test, '--', color='r', label='loss_test')    # 點和虛線
    plt.legend()
    plt.show()
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.plot(alpha, accuracy_train, '-', color='y', label='accuracy_train')    # 點和線
    plt.plot(alpha, accuracy_test, '--', color='y', label='accuracy_test')    # 點和虛線
    plt.legend()
    plt.show()
 
    
def multi_add(array1, array2, alpha):
    if type(array1) == list:
        for i in range(len(array1)):
            array1[i] = multi_add(array1[i], array2[i], alpha)
    elif type(array1) == np.ndarray:
        array1 = array1 * alpha + array2 * (1 - alpha)    
    return array1    

  
def add(array1, array2):
    if type(array1) == list:
        for i in range(len(array1)):
            array1[i] = add(array1[i], array2[i])
    elif type(array1) == np.ndarray:
        array1 = array1 + array2       
     
    
def multiple(array, alpha):
    print(type(array))
    if type(array) == list:
        for i in range(len(array)):
            array[i] = multiple(array[i], alpha)
    elif type(array) == np.ndarray:
        array = array * alpha


if __name__ == "__main__":
    main()    
