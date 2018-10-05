import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import collections
from mpl_toolkits.mplot3d import axes3d
import optimizer


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
      

def generate_data():
    x = np.linspace(0.00001, 1, 9999)
    y = np.sin(5*np.pi*x)/(5*np.pi*x)
    index = list(range(0, 9999))
    np.random.seed(10332)
    np.random.shuffle(index)
    x_u = x[index]
    y_u = y[index]
    x_train = x_u[0:9000]
    y_train = y_u[0:9000]
    x_test = x_u[9000:]
    y_test = y_u[9000:]
    return (x_train, y_train, x_test, y_test, x, y)


def model_generator():
    input = Input(shape=(1,), name='input')
    layer = Dense(5, activation='selu', name='layer1')(input)
    layer = Dense(5, activation='selu', name='layer2')(layer)
    layer = Dense(2, activation='selu', name='layer3')(layer)
    end = Dense(1, name='end')(layer)
    model = Model(input=input, output=end)
    model.compile(optimizer='adam', loss='mse')
    return model


def train(model, x_train, y_train, x_test, y_test, epochs):
    weights = [] 
    add_every_epochs = LambdaCallback(
        on_epoch_end=lambda epoch,
        logs: [weights.append([])]
    ) 
    save_weights = LambdaCallback(
        on_epoch_end=lambda epoch, 
        logs: [weights[-1].append(layer.get_weights()) for layer in model.layers] 
    ) 
    history = model.fit(x_train, y_train, batch_size=100,
                        epochs=epochs, verbose=1,
                        validation_data=(x_test, y_test))               
    return history, weights
    #### weights[epoch[layer, ...], ...] ####


def add_noise(weights):
    weight = weights.copy()
    if type(weight) == list:
        for i in range(len(weight)):
            weight[i] = add_noise(weight[i])
    else:
        weight = weight + np.random.normal(scale=np.abs(np.amax(weight)/10), size=weight.shape)
    return weight


def get_loss(model, weight, x, y):
    for i, layer in enumerate(model.layers):
        layer.set_weights(weight[i])    
    score = model.evaluate(x, y, batch_size=100, verbose=0)
    return score


def print_3D(weights, model, x, y, epochs):
    print('printing')
    loss = []
    tsne = TSNE(n_components=2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    weight = weights.copy()
    print('get loss and flatten')
    for i in range(len(weight)):
        loss.extend([get_loss(model, weight[i], x, y)])
        weight[i] = flatten(weight[i])
    print('tsne')
    x_y = tsne.fit_transform(weight)
    x, y = x_y.T
    loss = np.array(loss)
    ax.plot3D(x[:epochs], y[:epochs], loss[:epochs], 'gray')
    ax.scatter3D(x[epochs-1], y[epochs-1], loss[epochs-1], 'red')
    ax.scatter3D(x[epochs:], y[epochs:], loss[epochs:], c=loss[epochs:], cmap='CMRmap')
    plt.show()


epochs = 25
check_point = 100


def main():
    x_train, y_train, x_test, y_test, _, _ = generate_data()
    model = model_generator()
    _, weights = train(model, x_train, y_train, x_test, y_test, epochs)
    # weights[epoch[layer, ...], ...] ####
    for epoch in range(len(weights)):
        # for every epoch, check #check_point of point ####
        for i in range(check_point):
            weight_n = add_noise(weights[epoch])
            weights.append(weight_n)        
    print_3D(weights, model, x_test, y_test, epochs)


if __name__ == '__main__':
    main()

