import json
import numpy as np
import itertools
import re
import string

def data_generator(x_train_feature_dir, y_train_filename, thershold):
    thershold_of_occurences = thershold

    # Load data
    f = open(y_train_filename)
    print('start loading data')
    y_train = json.load(f)

    features = []
    captions = []
    num_of_captions = []

    possible_character = list(string.ascii_lowercase)

    for label_data in y_train:
        feature = np.load(x_train_feature_dir + '/feat/' + label_data['id']+'.npy')
        num_of_captions = len(label_data['caption'])
        for caption in label_data['caption']:
            features.append(feature)
            captions.append(re.sub(r'[^a-zA-Z0-9 ]', '', caption.lower()))


    print('start encoding')
    # init dictonary
    dic = {}
    word_to_idx = {
        'BOS': 0,
        'EOS': 1,
        'UWK': 2,
        'PAD': 3
    }
    next_word_id = 4
    # idx_to_word = None

    for caption in captions:
        for word in caption.split():
            dic[word] = dic.get(word, 0) + 1

    for item in dic.items():
        if item[1] > thershold_of_occurences:
            word_to_idx[item[0]] = next_word_id
            next_word_id += 1
        
    idx_to_word = dict((reversed(item) for item in word_to_idx.items()))

    captions = list(map(str.split, captions))

    features = np.array(features)
    captions = np.array(captions)

    max_length = max([len(caption) for caption in captions]) + 1
    sequence_length = np.array([len(caption) + 1 for caption in captions])

    y_inputs = np.array([[word_to_idx['BOS']] + y + [word_to_idx['PAD']] * (max_length - len(y) - 1) for y in captions])
    y_targets = np.array([y + [word_to_idx['EOS']] + [word_to_idx['PAD']] * (max_length - len(y) - 1) for y in captions])


    print('Done data generation!')
    return features, y_inputs, y_targets, word_to_idx, idx_to_word, next_word_id, max_length, sequence_length

def generate_batch(X, y_inputs, y_targets, word_idx, sequence_length, batch_size):

    idx = np.random.choice(len(X), batch_size)

    X_batch = X[idx]
    y_inputs_batch = y_inputs[idx]
    y_targets_batch = y_targets[idx]
    print(sequence_length)
    sequence_length_batch = sequence_length[idx]
    return X_batch, y_inputs_batch, y_targets_batch, sequence_length_batch


if __name__ == '__main__':
    X, y_inputs, y_targets, word_idx, idx_word, _, _, sequence_length = data_generator('./data/training_data', './data/training_label.json', 2)

    generate_batch(X, y_inputs, y_targets, word_idx, sequence_length, 128)