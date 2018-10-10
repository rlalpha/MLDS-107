#load_model
import keras.models
#data set
from keras.datasets import cifar10
from keras import utils
#basic
import numpy as np
#pic
import matplotlib.pyplot as plt


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
  

def generate_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    return (x_train, y_train, x_test, y_test)
  
  
def evaluate(model, x, y, batch_size=5000):
    loss, acc = model.evaluate(x[:batch_size], y[:batch_size], batch_size=batch_size)
    return loss, acc 
  

def get_weight(model):
    weight = []
    for layer in model.layers:
        weight.append(layer.get_weights())
    return weight
  
  
def add_noise(weights):
    weight = weights.copy()
    if type(weight) == list:
        for i in range(len(weight)):
            weight[i] = add_noise(weight[i])
    else:
        weight = weight + np.random.normal(scale=1e-5, size=weight.shape)
    return weight
  

def set_weight(model, weights):
    for i, layer in enumerate(model.layers):
        layer.set_weights(weights[i]) 
    return model    
  
  
def find_sharp(model, x, y, test_loss):
    print('find_sharpness')
    max_num = 0
    weights = get_weight(model)
    new_model = model_generator()
    for i in range(check_point):
        print('check:', i, 'points')
        new_weights = add_noise(weights)
        new_model = set_weight(new_model, new_weights)
        loss, _ = evaluate(new_model, x, y)
        if loss > max_num:
            max_num = loss 
    return (max_num - test_loss) / (1 + test_loss)        
            
            
check_point = 100 
        
        
def main():
    sharp = []
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    batch_size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] 
    batch_size = np.array(batch_size)
    batch_size_log = np.log10(batch_size)
    x_train, y_train, x_test, y_test = generate_data()
    for i in range(len(batch_size)):
        print('model:', batch_size[i], '_model.h5')
        model = keras.models.load_model(str(batch_size[i]) + '_model.h5')
        tr_l, tr_a = evaluate(model, x_train, y_train)
        train_loss.append(tr_l)
        train_acc.append(tr_a)
        te_l, te_a = evaluate(model, x_test, y_test)
        test_loss.append(te_l)
        test_acc.append(te_a)
        sh = find_sharp(model, x_test, y_test, te_l)
        sharp.append(sh)
        
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(batch_size_log, train_loss, '-', color='b', label='loss_train')
    plt.plot(batch_size_log, test_loss, '--', color='b', label='loss_test')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('batch_size(log)')
    ax2 = ax1.twinx()  # this is the important function
    plt.plot(batch_size_log, sharp, '-', color='r', label='sharp')
    ax2.set_ylabel('sharp')
    plt.legend()
    plt.show()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(batch_size_log, train_acc, '-', color='b', label='loss_train')
    plt.plot(batch_size_log, test_acc, '--', color='b', label='loss_test')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('batch_size(log)')
    ax2 = ax1.twinx()  # this is the important function
    plt.plot(batch_size_log, sharp, '-', color='r', label='sharp')
    ax2.set_ylabel('sharp')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
