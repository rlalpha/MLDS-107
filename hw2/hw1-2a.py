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
from matplotlib import pyplot as plt


layer_num = 3
epochs = 10
times = 8

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

    model.add( Conv2D( filters=48,kernel_size=(3,3) ) )
    model.add( Activation('relu') ) 
    
    model.add( Flatten() )

    model.add( Dense(10,activation='softmax')  )     
    model.compile( loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model
  
  
def train(x_train,y_train,x_test,y_test,epochs):
    weights = [[],[],[]]
    save_weights = LambdaCallback(
        on_epoch_end = lambda batch, 
        logs:[ weights[i].append(model.layers[2*i].get_weights()) for i in range (layer_num) ] 
    ) 
    model = model_generator()
    history = model.fit( x_train, y_train, batch_size=1000, epochs=epochs, verbose=1, validation_data=(x_test,y_test), callbacks = [save_weights] )
    return history, weights
  
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
  
def Pca(w, i):
    print('pca')
    pca = PCA(n_components=2)
    pca_data = []    
    for times in range(len(w)):
        layer = w[times]
        layer = reshape(layer)
        if times == 0:
            x = pca.fit_transform(layer[i])
        else:
            x = pca.transform(layer[i])
        pca_data.append(x)
    return pca_data

  
def Pca_w(w):
    print('pca_w')
    pca = PCA(n_components=2)
    pca_data = []
    for times in range(len(w)):
        data = []
        layer = w[times]
        for k in range(len(layer[0])):  #epochs_num
            data.append([])
            for j in range(len(layer)):  #layer_num
                data[k].append(layer[j][k])
            data[k] = flatten(data[k])
            
        if times == 0:
            x = pca.fit_transform(data)
        else:
            x = pca.transform(data)
        pca_data.append(x)
    return pca_data 
    
    
def print_xy(coordinate, loss):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(len(coordinate)):
        x_y = np.array(coordinate[i])
        x, y = x_y.T     
        plt.scatter(x, y, c=color[i], marker='.')
        for j in range (len(x)): 
            plt.text(x[j], y[j], round(loss[i][j],3), fontdict={'color':color[i]})
    plt.show()
    
    
def main():   
    x_train,y_train,x_test,y_test = generate_data()
    w = []
    his = []
    for i in range(times):
        print(i+1,'times')
        history, weights = train(x_train,y_train,x_test,y_test,epochs) #weights[ layer[ [epoch], [epoch]  ], layer [ [epoch], [epoch]  ]    ]
        w.append(weights)
        his.append(history.history['val_loss'])
    coor = Pca(w,0)
    print('layer0')
    print_xy(coor, his)
    coor2 = Pca_w(w)
    print('whole')
    print_xy(coor2, his)
    
    
if __name__ == '__main__':
    main()
  