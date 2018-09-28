import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

# visualizing graph arch
from keras.utils import plot_model

def generate_data_1():
    x = np.linspace(0.00001, 1, 9999)
    y = np.sin(5*np.pi*x)/(5*np.pi*x)

    index = list(range(0, 9999))
    np.random.seed(1024)
    np.random.shuffle(index)

    x_u = x[index]
    y_u = y[index]

    x_train = x_u[0:9000]
    y_train = y_u[0:9000]

    x_test = x_u[9000:]
    y_test = y_u[9000:]
    return (x_train, y_train, x_test, y_test, x, y)

def generate_data_2():
    x = np.linspace(0.00001, 1, 9999)
    y = np.sign(np.sin(5*np.pi*x))

    index = list(range(0, 9999))
    np.random.seed(1024)
    np.random.shuffle(index)

    x_u = x[index]
    y_u = y[index]

    x_train = x_u[0:9000]
    y_train = y_u[0:9000]

    x_test = x_u[9000:]
    y_test = y_u[9000:]
    return (x_train, y_train, x_test, y_test, x, y)


def model_generator_1():
    print('model_1 build')
    model = Sequential()
    model.add(Dense(5, input_shape=(1,)))
    model.add(Activation('relu'))

    model.add(Dense(5))
    model.add(Activation('relu'))

    model.add(Dense(5))
    model.add(Activation('relu'))

    model.add(Dense(5))
    model.add(Activation('relu'))

    model.add(Dense(5))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    return model


def model_generator_2():
    print('model_3 build')
    model = Sequential()
    model.add(Dense(10, input_shape=(1,)))
    model.add(Activation('relu'))

    model.add(Dense(6))
    model.add(Activation('relu'))

    model.add(Dense(6))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    return model


def model_generator_3():
    print('model_2 build')
    model = Sequential()
    model.add(Dense(45, input_shape=(1,)))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    return model


import matplotlib.pyplot as plt


def train(x_train, y_train, x_test, y_test, x, y, epochs, batch_size=100, functionName='default'):

    model_1 = model_generator_1()
    model_2 = model_generator_2()
    model_3 = model_generator_3()

    models = [model_1, model_2, model_3]
    for i in range(len(models)):
        plot_model(models[i], to_file='dnn_%s_model_%i.png'%(train.counter, i))

    history_1 = model_1.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
    history_2 = model_2.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
    history_3 = model_3.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))

    plt.plot(history_1.history['val_loss'],
             color='blueviolet', label='dnn-model-1-deep')
    plt.plot(history_2.history['val_loss'],
             color='brown', label='dnn-model-2-mediumn')
    plt.plot(history_3.history['val_loss'],
             color='green', label='dnn-model-1-shallow')
    plt.title("Val Loss Plot")
    plt.legend()
    # plt.show()
    plt.savefig('dnn_val_loss_%i_comparison_of_deep_and_shallow.png' %
                train.counter)
    plt.gcf().clear()

    y_1 = model_1.predict(x)
    y_2 = model_2.predict(x)
    y_3 = model_3.predict(x)

    plt.plot(y, color='black', label='ground_truth')
    plt.plot(y_1, color='blueviolet', label='dnn-model-1-deep')
    plt.plot(y_2, color='brown', label='dnn-model-2-mediumn')
    plt.plot(y_3, color='green', label='dnn-model-1-shallow')
    plt.title("Simulated Function %s Plot" % functionName)
    plt.legend()
    # plt.show()
    plt.savefig('dnn_simulation_%i_comparison_of_deep_and_shallow.png' %
                train.counter)
    plt.gcf().clear()

    train.counter += 1


def main():
    train.counter = 0
    x_train, y_train, x_test, y_test, x, y = generate_data_1()
    train(x_train, y_train, x_test, y_test, x, y, 50,
          functionName='sin(5*pi*x)/(5*pi*x)')
    x_train, y_train, x_test, y_test, x, y = generate_data_2()
    train(x_train, y_train, x_test, y_test, x, y, 150,
          functionName='sign(sin(5*pi*x))')


if __name__ == '__main__':
    main()
