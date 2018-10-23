import json
import numpy as np
import itertools

def data_generator(input_dir, label_file_path, thershold):

    # Loading the data
    print('starting load data......')
    f = open(label_file_path)
    reading_labels = json.load(f)
    features = []
    docs = []

    for data in reading_labels:
        feature = np.load(input_dir + '/feat/' + data['id']+'.npy')
        features.append(feature)
        docs.append(data['caption'][0].replace('.', ''))
    
    print('encoding data........')
    docs = [doc.split(" ") for doc in docs]

    dic = {}
    word_idx = {}
    idx_word = {}

    # count number of repeating words
    for doc in docs:
        for word in doc:
            dic[word] = dic.get(word, 0) + 1

    # token of BOS, EOS, UWK
    word_idx['BOS'] = 0
    word_idx['EOS'] = 1
    word_idx['UWK'] = 2
    word_idx['PAD'] = 3

    word_idx[0] = 'BOS'
    word_idx[1] = 'EOS'
    word_idx[2] = 'UWK'
    word_idx[3] = 'PAD'

    number_of_word = 4

    print('adding special symbol......')
    for key, cnt in dic.items():
        if cnt > thershold:
            word_idx[key] = number_of_word
            word_idx[number_of_word] = key
            number_of_word += 1
    
    docs = [[word_idx.get(word, 2) for word in doc] for doc in docs]
    
    features = np.array(features)
    docs = np.array(docs)

    max_length = max([len(doc) for doc in docs]) + 1
    sequence_length = np.array([len(doc) + 1 for doc in docs])

    y_inputs = np.array([[word_idx['BOS']] + y + [word_idx['PAD']] * (max_length - len(y) - 1) for y in docs])
    y_targets = np.array([y + [word_idx['EOS']] + [word_idx['PAD']] * (max_length - len(y) - 1) for y in docs])
    

    print(y_inputs, y_targets)
    print('Done data generation!')
    return features, y_inputs, y_targets, word_idx, idx_word, number_of_word, max_length, sequence_length

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