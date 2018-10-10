# basic
import numpy as np
# get dataset
from keras.datasets import mnist
from keras.utils import np_utils
# model structure
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
#pic
import matplotlib.pyplot as plt

def generate_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('shuffle')
    np.random.seed(1024)
    index = list(range(0, len(y_train)))
    np.random.shuffle(index)
    y_train = y_train[index]
    index = list(range(0, len(y_test)))
    np.random.shuffle(index)
    y_test = y_test[index]
    print('reshape')
    x_train = np.reshape(x_train,(60000,28,28,1))
    x_test = np.reshape(x_test,(10000,28,28,1))
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return(x_train, y_train, x_test, y_test)

               
def model_generator(activation='relu'):
    print('model build')
    input = Input(shape=(28,28,1,), name='input')
    layer = Conv2D(28, (3, 3), activation=activation, name='layer1')(input)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Flatten()(layer)
    end = Dense(10, activation='softmax', name='end')(layer)     
    model = Model(input=input, output=end)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

               
def train(model, x_train, y_train, x_test, y_test, epochs):
    history = model.fit(x_train, y_train, batch_size=500,
                        epochs=epochs, verbose=1,
                        validation_data=(x_test, y_test))    
    return history
               

epochs = 2000
               
               
def main():
    x_train, y_train, x_test, y_test = generate_data()
    model = model_generator()
    his = train(model, x_train, y_train, x_test, y_test, epochs)
    print(his.history)
    plt.plot(his.history['val_acc'], color="r", label='val_acc')
    plt.plot(his.history['acc'], color="b", label="acc")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
