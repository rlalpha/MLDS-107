import tensorflow as tf
import numpy as np
from alpha_preprocessing_data import generate_batch, data_generator

from seq import Seq2SeqModel


def get_train_data():

    x, y_inputs, y_targets, caption_id_to_feature_id, word_idx, \
        idx_word, num_of_words, max_length, sequence_lengths, video_id = \
        data_generator('./data/training_data', './data/training_label.json', 2)

    return x, y_inputs, y_targets, caption_id_to_feature_id, word_idx, \
        idx_word, num_of_words, max_length, sequence_lengths, video_id


def generate_model(x_train, y_train):
    pass


x, y_inputs, y_targets, caption_id_to_feature_id, word_idx, \
    idx_word, num_of_words, max_length, sequence_lengths, video_id \
    = get_train_data()

'''
ENCODER_INPUT_SIZE = 4096
HIDDEN_LAYER_SIZE = 1024
EMBEDDING_SIZE = 1024
NUM_OF_LAYER = 1
BOS = 0
EOS = 1
BATCH_SIZE = 100
KEEP_PROB = 0.7
'''

config = {
    'ENCODER_INPUT_SIZE': 4096,
    'HIDDEN_LAYER_SIZE': 1024,
    'EMBEDDING_SIZE': 1024,
    'NUM_OF_LAYER': 1,
    'BOS': 0,
    'EOS': 1,
    'BATCH_SIZE': 100,
    'KEEP_PROB': 0.7,

    'NUM_OF_WORDS': num_of_words,
    'MAX_LENGTH': max_length,

    'max_to_keep': False
}

model = Seq2SeqModel(config)

# training start
import math
model.sess.run(tf.global_variables_initializer())

epoc = 90
fake_max_sequence = np.array([config['MAX_LENGTH']] * config['BATCH_SIZE'])
for i in range(epoc):
    sample_prob_input = min(float(i) / epoc + 0.2, 1.0)
    for j in range(len(x) // config['BATCH_SIZE']):
        X_batch, y_inputs_batch, y_targets_batch, sequence_length_batch = generate_batch(x, y_inputs, y_targets, caption_id_to_feature_id, word_idx,
                                                                                         sequence_lengths, config['BATCH_SIZE'])
#         print(y_inputs_batch.shape, y_targets_batch.shape, max(sequence_length_batch))
        _, loss, prediction = model.train(X_batch, y_inputs_batch,
              y_targets_batch, sequence_length_batch, 
              fake_max_sequence, sample_prob_input)
    print([idx_word[idx] for idx in np.argmax(prediction[0], axis=1)])
    print('truth:', [idx_word[y_targets_batch[0, k]]
                     for k in range(max_length)])
    print("epoch {0}: loss : {1}".format(i, loss))
