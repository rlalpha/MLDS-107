import json
import numpy as np
import itertools
import string


x_train_feature_dir = './data/training_data'
y_train_filename = './data/training_label.json'

f = open(y_train_filename)
y_train = json.load(f)

features = []
captions = []
num_of_captions = []

possible_character = list(string.ascii_lowercase)
possible_character.extend(list(map(str, range(10))))
possible_character.extend([' ', ',', '.'])



strange_character_set = []
for label_data in y_train:
    feature = np.load(x_train_feature_dir + '/feat/' + label_data['id']+'.npy')
    num_of_captions = len(label_data['caption'])
    for caption in label_data['caption']:
        for char in caption.lower():
            if char not in possible_character:
                if char not in strange_character_set:
                    strange_character_set.append(char)
                    print(char)