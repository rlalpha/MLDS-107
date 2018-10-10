import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.callbacks import LambdaCallback
import numpy as np
from matplotlib import pyplot as plt 


def generate_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('reshape')
    x_train = np.reshape(x_train,(60000,28,28,1))
    x_test = np.reshape(x_test,(10000,28,28,1))
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (x_train,y_train,x_test,y_test)
  
  
def model_generator():
    print('model build')
    model = Sequential()
    model.add(Conv2D(filters=28, kernel_size=(3,3), input_shape=(28,28,1), padding='same')) 
    model.add(Activation('selu')) 
    model.add(Conv2D( filters=28,kernel_size=(3,3)))
    model.add(Activation('selu')) 
    model.add(Flatten())
    model.add(Dense(10,activation='softmax'))     
    model.compile( loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model
  

def train(x_train, y_train, x_test, y_test, epochs, batch_size):
    model = model_generator()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(x_test,y_test))
    model.save(str(batch_size) + "_model.h5")
    return history
 

epochs = 20


def main():   
    x_train,y_train,x_test,y_test = generate_data()
    history = train(x_train, y_train, x_test, y_test, epochs, 1000)
    history = train(x_train, y_train, x_test, y_test, epochs, 2000)
    
    
if __name__ == '__main__':
    main()
