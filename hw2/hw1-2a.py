import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
# get dataset
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.callbacks import LambdaCallback
import numpy as np
from sklearn.decomposition import PCA

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
    model.add( Conv2D( filters=28,kernel_size=(3,3),input_shape=(28,28,1),padding='same' ) ) 
    model.add( Activation('relu') ) 

    model.add( Conv2D( filters=28,kernel_size=(3,3) ) )
    model.add( Activation('relu') ) 

    model.add( Conv2D( filters=28,kernel_size=(3,3) ) )
    model.add( Activation('relu') ) 

    model.add( Conv2D( filters=28,kernel_size=(3,3) ) )
    model.add( Activation('relu') )
    
    model.add( Flatten() )

    model.add( Dense(10,activation='softmax')  )     
    model.compile( loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model
  
  
def train(x_train,y_train,x_test,y_test,epochs):
    weights = [[],[],[],[],[]]
    save_weights = LambdaCallback(on_epoch_end = lambda batch, \
                                  logs:[ weights[i].append(model.layers[2*i].get_weights()) for i in range (4) ] ) 
    model = model_generator()
    history = model.fit( x_train, y_train, batch_size=1000, epochs=epochs, verbose=1, validation_data=(x_test,y_test), callbacks = [save_weights] )
    return weights
  
import collections  
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
    
    
def reshape(layer):   
    for i in range (len(layer)):
        for j in range(len(layer[i])):
            layer[i][j] = flatten(layer[i][j])
    return layer
  
  

def Pca(layer):
    layer = reshape(layer)
    pca_data = []
    print('pca')  
    pca = PCA(n_components=2)
    x = pca.fit_transform(layer[0])
    pca_data.append(x)
    print(pca_data)    
            
     
    
def main():   
    x_train,y_train,x_test,y_test = generate_data()
    weights = train(x_train,y_train,x_test,y_test,5) #weights[ layer[ [epoch], [epoch]  ], layer [ [epoch], [epoch]  ]    ]
    Pca(weights)
    
    
if __name__ == '__main__':
    main()
