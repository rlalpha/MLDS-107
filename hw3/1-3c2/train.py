import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
#data set
from keras.datasets import cifar10
from keras import utils
#pic
import matplotlib.pyplot as plt
###### load data #########
!pip install -U -q PyDrive ## you will have install for every colab session
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from google.colab import files
# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


def generate_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    return (x_train, y_train, x_test, y_test)


def model_generator():
    input = Input(shape=(32, 32, 3,), name='input')
    layer = Conv2D(32, (3, 3), activation=activation, name='layer1')(input)
    layer = Conv2D(32, (3, 3), activation=activation, name='layer2')(layer)
    layer = Dropout(rate=0.2)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(64, (3, 3), activation=activation, name='layer3')(layer) 
    layer = Conv2D(64, (3, 3), activation=activation, name='layer4')(layer) 
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(128, (3, 3), activation=activation, name='layer5')(layer)
    layer = Conv2D(128, (3, 3), activation=activation, name='layer6')(layer)
    layer = Dropout(rate=0.5)(layer)
    layer = Flatten()(layer)
    end = Dense(10, activation='softmax', name='end')(layer)
    model = Model(input=input, output=end)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(model, x_train, y_train, x_test, y_test, epochs, batch_size):    
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, verbose=1,
                        validation_data=(x_test, y_test))                                     
    return model    


epochs = 40
activation = 'relu'


def main():
    x_train, y_train, x_test, y_test = generate_data()
    batch_size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] 
    for i in range(len(batch_size)):
        print('batch_size: ', batch_size[i])
        model = model_generator()
        model = train(model, x_train, y_train, x_test, y_test, epochs, batch_size=batch_size[i])
        model.save(str(batch_size[i]) + '_model.h5')
        #save to google drive
        uploaded = drive.CreateFile({'title': str(batch_size[i]) + '_model.h5'})
        uploaded.SetContentFile(str(batch_size[i]) + '_model.h5')
        uploaded.Upload()
 

def add_noise(weights):
    weight = weights.copy()
    if type(weight) == list:
        for i in range(len(weight)):
            weight[i] = add_noise(weight[i])
    else:
        weight = weight + np.random.normal(scale=np.abs(np.amax(weight)/10), size=weight.shape)
    return weight


def get_loss(model, x, y, batch_size=5000):   
    loss, acc = model.evaluate(x[:batch_size], y[:batch_size], batch_size=batch_size, verbose=0)
    return loss, acc
  

if __name__ == '__main__':
    main()
