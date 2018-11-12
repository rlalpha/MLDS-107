#load_model
import keras.models
#data set
from keras.datasets import cifar10
from keras import utils
#basic
import numpy as np
#pic
import matplotlib.pyplot as plt
##
import keras.backend as K
import keras.losses


def generate_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    return (x_train, y_train, x_test, y_test)
  

def get_sensitive(model, x, y, size=1000):
    x = x[:size]
    y = y[:size]
    y_label = K.placeholder(y.shape)
    loss = keras.losses.categorical_crossentropy(y_label, model.output)
    grad = K.gradients(loss, model.input)
    get_grads = K.function([model.input, y_label], grad)
    grads = get_grads([x ,y])[0]
    grads_norm  = np.sum(grads ** 2) ** 0.5
    return grads_norm
   
def evaluate(model, x, y, batch_size=5000):
    loss, acc = model.evaluate(x[:batch_size], y[:batch_size], batch_size=batch_size)
    return loss, acc 
    
    
def main():
    print('load')
    x_train, y_train, x_test, y_test = generate_data()
    print("start")
    sensitive = []
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    batch_size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    batch_size = np.array(batch_size)
    batch_size_log = np.log10(batch_size)
    for i in range(len(batch_size)):
        print('model:', batch_size[i], '_model.h5')
        model = keras.models.load_model(str(batch_size[i]) + '_model.h5')
        tr_l, tr_a = evaluate(model, x_train, y_train)
        train_loss.append(tr_l)
        train_acc.append(tr_a)
        te_l, te_a = evaluate(model, x_test, y_test)
        test_loss.append(te_l)
        test_acc.append(te_a)
        sensitive.append(get_sensitive(model, x_test, y_test))
        
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(batch_size_log, train_loss, '-', color='b', label='loss_train')
    plt.plot(batch_size_log, test_loss, '--', color='b', label='loss_test')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('batch_size(log)')
    plt.legend()
    ax2 = ax1.twinx()  # this is the important function
    plt.plot(batch_size_log, sensitive, '-', color='r')
    ax2.set_ylabel('sensitive')
    plt.legend()
    plt.show()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(batch_size_log, train_acc, '-', color='b', label='acc_train')
    plt.plot(batch_size_log, test_acc, '--', color='b', label='acc_test')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('batch_size(log)')
    plt.legend()
    ax2 = ax1.twinx()  # this is the important function
    plt.plot(batch_size_log, sensitive, '-', color='r')
    ax2.set_ylabel('sensitive')
    plt.legend()
    plt.show()
    
   
    
main()    

