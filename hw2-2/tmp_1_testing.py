import tensorflow as tf
import numpy as np
from dataset import generate_batch, data_generator, load_word2vec_dic, generate_embedding_matrix, generate_testing_data

from seq import Seq2SeqModel


def get_train_data():

    questions, y_inputs, y_targets, word_to_idx, idx_to_word, num_of_words, max_length, sequence_length = data_generator()

    return questions, y_inputs, y_targets, word_to_idx, \
        idx_to_word, num_of_words, max_length, sequence_length


def generate_model(x_train, y_train):
    pass



x, y_inputs, y_targets, word_to_idx, \
    idx_word, num_of_words, max_length, sequence_lengths \
    = get_train_data()
word_to_vec = load_word2vec_dic()
embedding_matrix = generate_embedding_matrix(word_to_vec, idx_word, num_of_words)

config = {
    'ENCODER_INPUT_SIZE': 4096,
    'HIDDEN_LAYER_SIZE': 1024,
    'EMBEDDING_SIZE': 1024,
    'NUM_OF_LAYER': 2,
    'BOS': 0,
    'EOS': 1,
    'BATCH_SIZE': 100,
    'KEEP_PROB': 0.7,

    'NUM_OF_WORDS': num_of_words,
    'MAX_LENGTH': max_length,

    'USING_ATTENTION': True,

    'max_to_keep': False,
    'embedding_matrix' : embedding_matrix
}

model = Seq2SeqModel(config)

# training start
import math
saver = tf.train.Saver()
# model.sess.run(tf.global_variables_initializer())

# epoc = 100
# for i in range(epoc):
#     sample_prob_input = min(float(i) / epoc + 0.2, 1.0)
#     for j in range(len(x) // config['BATCH_SIZE']):
#         x_batch, y_inputs_batch, y_targets_batch, sequence_length_batch = \
#             generate_batch(x, y_inputs, y_targets,
#                            word_to_idx, sequence_lengths, batch_size=config['BATCH_SIZE'])
#         _, loss, prediction = model.train(x_batch, y_inputs_batch,
#                                           y_targets_batch, sequence_length_batch,
#                                           fake_max_sequence, sample_prob_input)
#         if j % 20 == 0:
#             print([idx_word[idx] for idx in np.argmax(prediction[0], axis=1)])
#             print('truth:', [idx_word[y_targets_batch[0, k]]
#                         for k in range(max_length)])
#             print("epoch {0}: loss : {1}".format(i, loss))

# load ing result
# saver.restore(model.sess, "/tmp/model.ckpt")

testing_word = generate_testing_data(word_to_idx)
from sklearn.model_selection import KFold

kf = KFold(n_splits=100)

result = []
for train_index, test_index in kf.split(testing_word):
    X_test = testing_word[test_index]
    fake_max_sequence = np.array([config['MAX_LENGTH']] * len(X_test))
    prediction = model.predict(X_test, sequence_lengths, fake_max_sequence)
    prediction = [ [ idx_word[idx] for idx in np.argmax(prediction[i], axis = 1) ] for i in range(X_test.shape[0]) ]
    prediction = [[ word for word in pred if (word != 'EOS' and word != 'BOS' and word != 'UWK')] for pred in prediction]
    prediction = [ "".join(data) for data in prediction]
    result.append(prediction)
    print(prediction)

with open('output_trial.txt', 'w') as f:
    for item in result:
        f.write("%s\n" % item)