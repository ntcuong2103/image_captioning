import pickle
import random

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, to_categorical


def get_tokenizer(desc_list):
    tokenizer = Tokenizer(
        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', oov_token="<unk>"
    )
    tokenizer.fit_on_texts(desc_list)
    filtered_vocab = [w for w, c in tokenizer.word_counts.items() if c >= 5]
    tokenizer.word_index = {
        w: id
        for w, id in tokenizer.word_index.items()
        if w in filtered_vocab + ["<unk>"]
    }
    num_vocab = len(tokenizer.word_index)
    tokenizer.word_index.update({"<s>": num_vocab + 1, "<e>": num_vocab + 2})
    return tokenizer


class DataGenerator(Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        list_IDs,
        labels,
        n_classes,
        tokenizer,
        max_length,
        batch_size=1,
        shuffle=True,
    ):
        "Initialization"
        self.img_features = pickle.load(open("img_features.pkl", "rb"))
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y_in = []
        y_out = []

        # Generate data
        for ID in list_IDs_temp:
            X.append(self.img_features[ID][0])
            y = random.choice(self.labels[ID])
            y_in.append("<s> " + y)
            y_out.append(y + " <e>")

        return [
            np.array(X),
            pad_sequences(
                self.tokenizer.texts_to_sequences(y_in),
                padding="post",
                maxlen=self.max_length,
            ),
        ], to_categorical(
            pad_sequences(
                self.tokenizer.texts_to_sequences(y_out),
                padding="post",
                maxlen=self.max_length,
            ),
            num_classes=self.n_classes,
        )


def main():
    train_ids = [
        line.strip()
        for line in open("Flickr8k_text/Flickr_8k.trainImages.txt").readlines()
    ]
    from collections import defaultdict
    from itertools import chain

    annotations = defaultdict(list)
    for line in open("Flickr8k_text/Flickr8k.token.txt").readlines():
        if len(line.strip().split("\t")) == 2:
            annotations[line.strip().split("\t")[0][:-2]].append(
                line.strip().split("\t")[1]
            )

    desc_list = list(chain(*[annotations[id] for id in train_ids]))
    tokenizer = get_tokenizer(desc_list)
    max_length = max(map(len, desc_list)) + 1
    # padding for sequence: <pad> -> 0
    data_gen = DataGenerator(
        train_ids,
        annotations,
        len(tokenizer.word_index) + 1,
        tokenizer,
        max_length,
        batch_size=5,
    )


# main()
