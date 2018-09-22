import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Activation

x = np.linspace(0.00001,1,9999)
y = np.sin(5*np.pi*x)/(5*np.pi*x)

index=list(range(0,9999))
np.random.seed(1024)
np.random.shuffle(index)

x = x[index]
y = y[index]

x_train = x[0:9000]
y_train = y[0:9000]

x_test = x[9000:]
y_test = y[9000:]

#### generate model ####
print('model build')

model = Sequential()
model.add( Dense( 5 , input_shape=(1,) ) )
model.add( Activation('relu') ) 

model.add( Dense(5) )
model.add( Activation('relu') ) 

model.add( Dense(5) )
model.add( Activation('relu') ) 

model.add( Dense(5) )
model.add( Activation('relu') ) 

model.add( Dense(1) )
model.compile( loss = 'mse', optimizer='adam',metrics=['accuracy'])
#### generate model ####

#### train ####
print('start training')
history = model.fit( x_train, y_train, batch_size=100, epochs=100, verbose=1, validation_data=(x_test,y_test) )
 
#### save model ####
print('save file')
model.save('hw1-1a.h5')
    
import matplotlib.pyplot as plt          
#### print accuracy ####
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) 