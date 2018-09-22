import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D

# get dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('reshape')
x_train = np.reshape(x_train,(60000,28,28,1))
x_test = np.reshape(x_test,(10000,28,28,1))
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#### generate model ####  
print('model build')
model = Sequential()
model.add( Conv2D( filters=28,kernel_size=(3,3),input_shape=(28,28,1),padding='same' ) ) 
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
#### generate model ####
          
#### train ####
print('start training')
history = model.fit( x_train, y_train, batch_size=1000, epochs=10, verbose=1, validation_data=(x_test,y_test) )
 
#### save model ####
print('save file')
model.save('hw1-1b.h5')
    
import matplotlib.pyplot as plt          
#### print accuracy ####
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc']) 