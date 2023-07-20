from data import get_tokenizer, DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from model import Concat_LSTM_GloVe, Concat_LSTM, Add_LSTM, Attention_LSTM
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm

import tensorflow as tf

config = tf.compat.v1.ConfigProto() 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
  
# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
 
def generate_desc(model, tokenizer, img_feature, max_length):
    in_text = '<s>'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])
        sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
        pred = model.predict([img_feature,sequence], verbose=0)
        pred = np.argmax(pred, -1)[0][i]
        if pred not in [tokenizer.word_index['<e>'], 0]:
            word = tokenizer.index_word[pred]
            in_text += ' ' + word
        else: 
            break
    return ' '.join(in_text.split(' ')[1:])

def evaluate_model(model, test_ids, annotations, image_features, tokenizer, max_length):
    actual, predicted, blues = list(), list(), list()
    # step over the whole set
    for key in tqdm(test_ids):
        desc_list = annotations[key]
        # generate description
        yhat = generate_desc(model, tokenizer, image_features[key], max_length)
        # store actual and predicted
        references = [d.split() for d in tokenizer.sequences_to_texts(tokenizer.texts_to_sequences(desc_list))]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

if __name__ == '__main__':
    train_ids = [line.strip() for line in open('Flickr8k_text/Flickr_8k.trainImages.txt').readlines()]
    test_ids = [line.strip() for line in open('Flickr8k_text/Flickr_8k.testImages.txt').readlines()]
    from collections import defaultdict
    from itertools import chain
    annotations = defaultdict(list)
    for line in open('Flickr8k_text/Flickr8k.token.txt').readlines():
        if len(line.strip().split('\t')) == 2:
            annotations[line.strip().split('\t')[0][:-2]].append(line.strip().split('\t')[1])

    desc_list = list(chain(*[annotations[id] for id in train_ids]))
    tokenizer = get_tokenizer(desc_list)
    max_length = max(map(len, desc_list)) + 1
    # padding for sequence: <pad> -> 0
    import pickle
    embedding_matrix = pickle.load(open('embedding.pkl', 'rb'))
    model = Attention_LSTM(len(tokenizer.word_index) + 1, 300, max_length, 0.5)
    model.load_weights('models/attention_LSTM/weights.16-0.37.hdf5')
    batch_size = 1

    # generator_test = DataGenerator(test_ids, annotations, len(tokenizer.word_index) + 1, tokenizer, max_length, batch_size=batch_size, shuffle=False)

    # [X, y_in], y_out = generator_test.__getitem__(0)
    # desc = generate_desc(model, tokenizer, np.array([X[0]]), 200)
    import pickle
    # evaluate_model(model, test_ids, annotations, pickle.load(open('models/attention_LSTM/img_features_2d.pkl', 'rb')), tokenizer, 200)
    
    evaluate_model(model, ['3544673666_ffc7483c96.jpg'], annotations, pickle.load(open('models/attention_LSTM/img_features_2d.pkl', 'rb')), tokenizer, 200)
    pass