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
    def __init__(self, sess, z_d = 1000, batch_size = 32, epochs = 20, log_path = "./log"):
        self.sess = sess
        self.z_d = z_d
        self.output_dimension = (64, 64, 3)
        self.batch_size = batch_size

        file_list = load_image_file_list()
        self.batch_num = int(len(file_list) / batch_size)

        # generator for generated image
        self.z = tf.placeholder(dtype = tf.float32, shape = (None, z_d))
        self.generated_img = self.generator(self.z)

        # discriminator for real image
        iterator = load_animation_face_iterator(file_list, epochs = epochs)
        real_img = iterator.get_next()
        real_img = tf.reshape(real_img, [-1, self.output_dimension[0],
        self.output_dimension[1], self.output_dimension[2]])
        print(real_img)

        # score and loss
        self.score_real = self.discriminator(real_img)
        self.score_fake = self.discriminator(self.generated_img, reuse = True)

        self.D_loss = tf.reduce_mean(self.score_real) - tf.reduce_mean(self.score_fake)
        self.G_loss = - tf.reduce_mean(self.score_fake)

        # collection of variable
        D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

        # train_op
        self.D_train_op = tf.train.RMSPropOptimizer(1e-4).minimize(-self.D_loss, var_list = D_vars)
        self.G_train_op = tf.train.RMSPropOptimizer(1e-4).minimize(self.G_loss, var_list = G_vars)

        # clip operation
        self.clip_op = [ p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in D_vars]

        # define summary
        tf.summary.scalar('D_loss', self.D_loss)
        tf.summary.scalar('G_loss', self.G_loss)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(log_path,
                                      sess.graph)

        sess.run(tf.global_variables_initializer())


    def discriminator(self, img, scope = 'discriminator', reuse = False):

        with tf.variable_scope(scope, reuse = reuse):

            # layer 1 None, 64, 64, 3 --> None, 32, 32, 50
            Z = conv2d(img, filters = 50, kernel_size = (4, 4), strides = (2, 2)
            , activation = tf.nn.leaky_relu, padding = "same")
            Z = batch_normalization(Z)
            print(Z)

            # layer 2 None, 32 ,32, 50 --> None, 16, 16, 25
            Z = conv2d(Z, filters = 25, kernel_size = (4, 4), strides = (2, 2)
            , activation = tf.nn.leaky_relu, padding = "same")
            Z = batch_normalization(Z)
            print(Z)

            # layer 3 None, 16 ,16, 25 --> None, 16, 16, 12
            Z = conv2d(Z, filters = 12, kernel_size = (4, 4), strides = (1, 1)
            , activation = tf.nn.leaky_relu, padding = "same")
            Z = batch_normalization(Z)
            print(Z)

            # Dense Layer
            Z = flatten(Z)
            Z = dense(Z, 512)
            Z = dense(Z, 128)
            Z = dense(Z, 1)
            print(Z)
            
            return Z
        
    def generator(self, z, scope = 'generator'):

        with tf.variable_scope(scope):

            input_img = dense(z, 3072)
            input_img = tf.reshape(input_img, shape = [-1, 16, 16, 12])

            # layer 1 None, 16, 16, 12 --> None, 32, 32, 25
            Z = conv2d_transpose(input_img, filters = 25, kernel_size = (4, 4), strides = (2, 2),
            activation = tf.nn.leaky_relu, padding = "same")
            Z = batch_normalization(Z)
            print(Z)

            # layer 2 None, 32, 32, 25 --> None, 64, 64, 50
            Z = conv2d_transpose(Z, filters = 50, kernel_size = (4, 4), strides = (2, 2),
            activation = tf.nn.leaky_relu, padding = "same")
            Z = batch_normalization(Z)
            print(Z)

            
            Z = conv2d_transpose(Z, filters = 3, kernel_size = (4, 4), strides = (1, 1),
            activation = tf.nn.sigmoid, padding = "same")
            Z = batch_normalization(Z)
            print(Z)

            return Z

    def sample_z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def train_D(self):
        z = self.sample_z(self.batch_size, self.z_d)
        _, loss, _ = self.sess.run([self.D_train_op, self.D_loss, self.clip_op], feed_dict = {self.z : z})
        return loss
    
    def train_G(self):
        z = self.sample_z(self.batch_size, self.z_d)
        _, loss, summary = self.sess.run([self.G_train_op, self.G_loss, self.merged], feed_dict = {self.z : z})
        self.train_writer.add_summary(summary)
        return loss
    
    def generate_testing_img(self):
        z = self.sample_z(self.batch_size, self.z_d)
        fake_img = self.sess.run(self.generated_img, feed_dict = {self.z : z})
        return fake_img

def plot(samples):
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(4, 8)
    gs.update(wspace=0.00, hspace=0.00)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(64, 64, 3))

    return fig

if __name__ == "__main__":

    sess = tf.Session()
    EPOCHS = 10
    model = WGAN(sess, epochs = EPOCHS)
    
    for i in range(EPOCHS):
        generated = model.generate_testing_img()
        fig = plot(generated)
        plt.show(fig)

        for j in range(model.batch_num):

            D_loss = model.train_D()

            G_loss = model.train_G()

        print ('D_loss:', D_loss, 'G_loss', G_loss)
        

