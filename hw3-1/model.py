from data_loader import load_image_file_list, load_animation_face_iterator
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.layers import conv2d, max_pooling2d, batch_normalization, dense, flatten, conv2d_transpose

class WGAN(object):
    
    # def init
    def __init__(self, sess, z_d = 100, batch_size = 64, epochs = 20, log_path = "./log"):
        self.sess = sess
        self.z_d = z_d
        self.output_dimension = (64, 64, 3)
        self.batch_size = batch_size
        self.is_training = tf.placeholder(tf.bool, shape = ())

        file_list = load_image_file_list()
        self.batch_num = int(len(file_list) / batch_size)

        # generator for generated image
        self.z = tf.placeholder(dtype = tf.float32, shape = (None, z_d))
        self.generated_img = self.generator(self.z)

        # discriminator for real image
        iterator = load_animation_face_iterator(file_list, epochs = epochs + 20, batch_size = batch_size)
        self.real_img = iterator.get_next()
        self.real_img = tf.reshape(self.real_img, [-1, self.output_dimension[0],
        self.output_dimension[1], self.output_dimension[2]])

        # score and loss
        self.score_real = self.discriminator(self.real_img)
        self.score_fake = self.discriminator(self.generated_img, reuse = True)

        self.D_loss = - tf.reduce_mean(self.score_real) + tf.reduce_mean(self.score_fake)
        self.G_loss = - tf.reduce_mean(self.score_fake)

        # collection of variable
        D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

        # train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.D_train_op = tf.train.RMSPropOptimizer(- 2e-4).minimize(self.D_loss, var_list = D_vars)
            self.G_train_op = tf.train.RMSPropOptimizer(- 2e-4).minimize(self.G_loss, var_list = G_vars)


        # clip operation
        self.clip_op = [ p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in D_vars]

        # define summary
        tf.summary.scalar('D_loss', self.D_loss)
        tf.summary.scalar('G_loss', self.G_loss)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(log_path,
                                      sess.graph)

        sess.run(tf.global_variables_initializer())


    def discriminator(self, img, scope = 'discriminator', ndf = 64, reuse = False):

        with tf.variable_scope(scope, reuse = reuse):

            # layer 1 None, 64, 64, 3 --> None, 32, 32, ndf
            Z = conv2d(img, filters = ndf, kernel_size = (5, 5), strides = (2, 2)
            , activation = tf.nn.leaky_relu, padding = "same", kernel_initializer = tf.random_normal_initializer(stddev=0.02))
            print(Z)

            # layer 2 None, 32 ,32, ndf --> None, 16, 16, ndf * 2
            Z = conv2d(Z, filters = ndf * 2, kernel_size = (5, 5), strides = (2, 2)
            , activation = tf.nn.leaky_relu, padding = "same", kernel_initializer = tf.random_normal_initializer(stddev=0.02))
            Z = batch_normalization(Z, training = self.is_training)
            print(Z)

            # layer 3 None, 16 ,16, ndf * 2 --> None, 8, 8, ndf * 4
            Z = conv2d(Z, filters = ndf * 4, kernel_size = (5, 5), strides = (2, 2)
            , activation = tf.nn.leaky_relu, padding = "same", kernel_initializer = tf.random_normal_initializer(stddev=0.02))
            Z = batch_normalization(Z, training = self.is_training)
            print(Z)

            # layer 4 None, 8 ,8, ndf * 4 --> None, 4, 4, ndf * 8
            Z = conv2d(Z, filters = ndf * 8, kernel_size = (5, 5), strides = (2, 2)
            , activation = tf.nn.leaky_relu, padding = "same", kernel_initializer = tf.random_normal_initializer(stddev=0.02))
            Z = batch_normalization(Z, training = self.is_training)
            print(Z)

            # Dense Layer
            Z = flatten(Z)
            Z = dense(Z, 1)
            print(Z)
            
            return Z
        
    def generator(self, z, scope = 'generator', ngf = 64):

        with tf.variable_scope(scope):

            input_img = dense(z, 4 * 4 * 8 * ngf, activation = tf.nn.leaky_relu)
            input_img = batch_normalization(input_img)
            input_img = tf.reshape(input_img, shape = [-1, 4, 4, 8 * ngf])

            # layer 1 None, 4, 4, 8 * ngf --> None, 8, 8, 4 * ngf
            Z = conv2d_transpose(input_img, filters = 4 * ngf, kernel_size = (5, 5), strides = (2, 2),
            activation = tf.nn.leaky_relu, padding = "same", kernel_initializer = tf.random_normal_initializer(stddev=0.02))
            Z = batch_normalization(Z, training = self.is_training)
            print(Z)

            # layer 2 None, 8, 8, 4 * ngf --> None, 16, 16, 2 * ngf
            Z = conv2d_transpose(Z, filters = 2 * ngf, kernel_size = (5, 5), strides = (2, 2),
            activation = tf.nn.leaky_relu, padding = "same", kernel_initializer = tf.random_normal_initializer(stddev=0.02))
            Z = batch_normalization(Z, training = self.is_training)
            print(Z)

            # layer 3 None, 16, 16, 2 * ngf --> None, 32, 32, ngf
            Z = conv2d_transpose(Z, filters = ngf, kernel_size = (5, 5), strides = (2, 2),
            activation = tf.nn.leaky_relu, padding = "same", kernel_initializer = tf.random_normal_initializer(stddev=0.02))
            Z = batch_normalization(Z, training = self.is_training)
            print(Z)

            # layer 4 None, 32, 32, ngf --> None, 64, 64, 3
            Z = conv2d_transpose(Z, filters = 3, kernel_size = (5, 5), strides = (2, 2),
            activation = tf.nn.tanh, padding = "same", kernel_initializer = tf.random_normal_initializer(stddev=0.02))
            print(Z)

            return Z

    def sample_z(self, m, n):
        return np.random.normal(size=[m, n])

    def train_D(self):
        z = self.sample_z(self.batch_size, self.z_d)
        _, loss, _ = self.sess.run([self.D_train_op, self.D_loss, self.clip_op], feed_dict = {self.z : z, self.is_training : True})
        return loss
    
    def train_G(self):
        z = self.sample_z(self.batch_size, self.z_d)
        _, loss, summary = self.sess.run([self.G_train_op, self.G_loss, self.merged], feed_dict = {self.z : z, self.is_training : True})
        self.train_writer.add_summary(summary)
        return loss
    
    def generate_testing_img(self):
        z = self.sample_z(self.batch_size, self.z_d)
        fake_img = self.sess.run(self.generated_img, feed_dict = {self.z : z, self.is_training : False})
        return fake_img
    
    def generate_real_img(self):
        real_img = self.sess.run(self.real_img)
        return real_img

def plot(samples):
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.00, hspace=0.00)

    for i, sample in enumerate(samples):
        sample = (sample + 1.0) / 2.0
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(64, 64, 3))

    return fig

if __name__ == "__main__":

    sess = tf.Session()
    EPOCHS = 20000
    model = WGAN(sess, epochs = EPOCHS)
    
    for i in range(EPOCHS):
        if i == 0:
            real = model.generate_real_img()
            fig = plot(real)
            fig.savefig('./result/real.png')
            plt.close(fig)

        for j in range(model.batch_num):

            if i < 1 and j < 25:
                n_c = 100
            else:
                n_c = 5
            
            for k in range(n_c):
                D_loss = model.train_D()

            G_loss = model.train_G()

            if j % 50 == 0:
                print (i, j, 'D_loss:', D_loss, 'G_loss', G_loss)
                generated = model.generate_testing_img()
                fig = plot(generated)
                fig.savefig('./result/'+ str(i) + '_' + str(j) + '.png')   # save the figure to file
                plt.close(fig)    # close the figure

        

