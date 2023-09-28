import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from beamsearch import generate_sequence_beamsearch
from data import get_tokenizer
from model import Merge_LSTM, Attention_LSTM, Injection_LSTM

config = tf.compat.v1.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def generate_desc(model, tokenizer, img_feature, max_length):
    in_text = "<s>"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])
        sequence = pad_sequences(sequence, maxlen=max_length, padding="post")
        pred = model.predict([img_feature, sequence], verbose=0)
        pred = np.argmax(pred, -1)[0][i]
        if pred not in [tokenizer.word_index["<e>"], 0]:
            word = tokenizer.index_word[pred]
            in_text += " " + word
        else:
            break
    return " ".join(in_text.split(" ")[1:])


def evaluate_model(
    model, test_ids, annotations, image_features, tokenizer, max_length
):
    actual, predicted, blues = list(), list(), list()
    # step over the whole set
    for key in tqdm(test_ids):
        desc_list = annotations[key]
        # generate description
        # yhat = generate_desc(model, tokenizer, image_features[key], max_length)
        yhat = generate_sequence_beamsearch(
            lambda prefixes: model.predict(
                [
                    np.repeat(image_features[key], len(prefixes), axis=0),
                    pad_sequences(prefixes, maxlen=max_length, padding="post"),
                ],
                verbose=0,
            )[
                np.arange(len(prefixes)),
                [len(prefix) - 1 for prefix in prefixes],
            ],
            tokenizer,
        )
        # store actual and predicted
        references = [
            d.split()
            for d in tokenizer.sequences_to_texts(
                tokenizer.texts_to_sequences(desc_list)
            )
        ]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print(
        "BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    )
    print(
        "BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    )
    print(
        "BLEU-3: %f"
        % corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    )
    print(
        "BLEU-4: %f"
        % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    )


if __name__ == "__main__":
    train_ids = [
        line.strip()
        for line in open("Flickr8k_text/Flickr_8k.trainImages.txt").readlines()
    ]
    test_ids = [
        line.strip()
        for line in open("Flickr8k_text/Flickr_8k.testImages.txt").readlines()
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
    import pickle

    embedding_matrix = pickle.load(open("embedding.pkl", "rb"))

    model = Injection_LSTM(
        1536,
        128,
        vocab_size=len(tokenizer.word_index) + 1,
        embedding_dim=300,
        embedding_matrix=embedding_matrix,
        max_length=max_length,
        dropout=0.5,
    )

    model.load_weights("models/injection_inception_w2v_128/weights.46-0.40.hdf5")
    batch_size = 1

    import pickle

    evaluate_model(
        model,
        test_ids,
        annotations,
        pickle.load(open("models/encoded_features/inception_resnet_v2.pkl", "rb")),
        tokenizer,
        200,
    )

    # evaluate a single image
    # evaluate_model(model, ['3484649669_7bfe62080b.jpg', '3544673666_ffc7483c96.jpg', '2399219552_bbba0a9a59.jpg', '3514184232_b336414040.jpg'], annotations, pickle.load(open('models/encoded_features/inception_resnet_v2_avg.pkl', 'rb')), tokenizer, 200)
