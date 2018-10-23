import tensorflow as tf
from model.seq2seq import Seq2seq
from preposessing_data import generate_batch, data_generator

BATCH_SIZE = 128
ENCODE_INPUT_SIZE = 4096
HIDDEN_UNIT_SIZE = 1024
EMEDDING_SIZE = 1024
if __name__ == '__main__':
    sess = tf.InteractiveSession()

    X, y_inputs, y_targets, word_idx, idx_word, num_of_words, _, sequence_length = data_generator('./data/training_data', './data/training_label.json', 2)
    model = Seq2seq(sess, ENCODE_INPUT_SIZE, HIDDEN_UNIT_SIZE, EMEDDING_SIZE, num_of_words)
    sess.run(tf.global_variables_initializer())

    epoc = 20

    for i in range(epoc):

        for j in range(len(X) // BATCH_SIZE):
            X_batch, y_inputs_batch, y_targets_batch, sequence_length_batch = generate_batch(X, y_inputs, y_targets, word_idx,
             sequence_length, BATCH_SIZE)

            model.trian(X_batch, y_inputs_batch, y_targets_batch, sequence_length_batch)


            
