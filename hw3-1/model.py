from data_loader import load_image_file_list, load_animation_face_iterator
import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, batch_normalization, dense, flatten, conv2d_transpose

class WGAN(object):
    
    # def init
    def __init__(self, sess, z_d = 1000):
        self.sess = sess
        self.z_d = z_d
        self.output_dimension = (64, 64, 3)

        file_list = load_image_file_list()

        # generator for generated image
        z = tf.placeholder(dtype = tf.float32, shape = (None, z_d))
        generated_img = self.generator(z)

        # discriminator for real image
        real_img = load_animation_face_iterator(file_list).get_next()
        real_img = tf.reshape(real_img, [-1, self.output_dimension[0],
        self.output_dimension[1], self.output_dimension[2]])
        print(real_img)

        # score and loss
        self.score_real = self.discriminator(real_img)
        self.score_fake = self.discriminator(generated_img, reuse = True)

        self.D_loss = tf.reduce_mean(self.score_real) - tf.reduce_mean(self.score_fake)
        self.G_loss = - tf.reduce_mean(self.score_real)
    

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
        
    def generator(self, z):
        
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
        activation = tf.nn.leaky_relu, padding = "same")
        Z = batch_normalization(Z)
        print(Z)

        return Z

if __name__ == "__main__":

    sess = tf.Session()

    model = WGAN(sess)

