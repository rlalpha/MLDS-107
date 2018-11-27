from PIL import Image
import numpy as np
import glob
import tensorflow as tf
from tensorflow.data import Dataset, Iterator


def load_image_file_list( dir = './faces' ) :
    file_list = glob.glob(dir + '/*.jpg')
    return file_list

def load_animation_face_iterator(file_list, epochs = 10, resized_img = (64, 64), batch_size = 32):

    # loading the file name
    img_path = Dataset.from_tensor_slices(file_list)

    def load_image(img_path):
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3, dtype= tf.float32) / 255
        img_decoded = tf.image.resize_image_with_crop_or_pad(img_decoded, resized_img[0], resized_img[1])

        return img_decoded
    
    # mapping the image
    img_decoded = img_path.map(load_image)
    img_decoded = img_decoded.repeat(epochs)
    img_decoded = img_decoded.batch(batch_size)
    dataset_iterator = img_decoded.make_one_shot_iterator()

    return dataset_iterator

if __name__ == "__main__":
    file_list = load_image_file_list()
    print(load_animation_face_iterator(file_list))