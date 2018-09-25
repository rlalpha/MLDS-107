
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D

# get dataset
from keras.datasets import mnist


def generate_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('reshape')
    x_train = np.reshape(x_train,(60000,28,28,1))
    x_test = np.reshape(x_test,(10000,28,28,1))
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (x_train,y_train,x_test,y_test)

def model_generator_1():
    print('model build')
    model = Sequential()
    model.add( Conv2D( filters=28,kernel_size=(3,3),input_shape=(28,28,1),padding='same' ) ) 
    model.add( Activation('relu') ) 

    model.add( Conv2D( filters=28,kernel_size=(3,3) ) )
    model.add( Activation('relu') ) 

    model.add( Conv2D( filters=28,kernel_size=(3,3) ) )
    model.add( Activation('relu') ) 
    
    model.add( Conv2D( filters=28,kernel_size=(3,3) ) )
    model.add( Activation('relu') )
    
    model.add( Conv2D( filters=28,kernel_size=(3,3) ) )
    model.add( Activation('relu') )

    model.add( Flatten() )

    model.add( Dense(1024) )    
    model.add(Activation('relu'))

    model.add( Dense(10,activation='softmax')  )     
    model.compile( loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    print(model.summary())
    return model

def model_generator_2():
    print('model build')
    model = Sequential()
    model.add( Conv2D( filters=28,kernel_size=(5,5),input_shape=(28,28,1),padding='same' ) ) 
    model.add( Activation('relu') ) 

    model.add( Conv2D( filters=28,kernel_size=(5,5) ) )
    model.add( Activation('relu') ) 

    model.add( Conv2D( filters=28,kernel_size=(5,5) ) )
    model.add( Activation('relu') ) 
    
    model.add( Flatten() )

    model.add( Dense(1024) )    
    model.add(Activation('relu'))

    model.add( Dense(10,activation='softmax')  )     
    model.compile( loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    print(model.summary())
    return model
  


import matplotlib.pyplot as plt     
def train(x_train,y_train,x_test,y_test,epochs):
    
    model_1 = model_generator_1()
    model_2 = model_generator_2()
    
    history_1 = model_1.fit( x_train, y_train, batch_size=1000, epochs=epochs, verbose=1, validation_data=(x_test,y_test) )
    history_2 = model_2.fit( x_train, y_train, batch_size=1000, epochs=epochs, verbose=1, validation_data=(x_test,y_test) )
    
    
    plt.plot(history_1.history['val_loss'], color = 'blueviolet', label='cnn-model-1-deep')
    plt.plot(history_2.history['val_loss'], color = 'blue', label='cnn-model-1-shallow')
    plt.title("Val Loss Plot")
    plt.legend()
    plt.show()     
    
    plt.plot(history_1.history['val_acc'], color = 'blueviolet', label='cnn-model-1-deep')
    plt.plot(history_2.history['val_acc'], color = 'blue', label='cnn-model-2-shallow')
    plt.title("Val Acc Plot")
    plt.legend()
    plt.show() 
    
def main():   
    x_train,y_train,x_test,y_test = generate_data()
    train(x_train,y_train,x_test,y_test,10)
    
    
if __name__ == '__main__':
    main()
