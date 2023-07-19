from data import get_tokenizer, DataGenerator
from model import Concat_LSTM_GloVe
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from embedding import get_embedding
import tensorflow as tf
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

if __name__ == '__main__':
    train_ids = [line.strip() for line in open('Flickr8k_text/Flickr_8k.trainImages.txt').readlines()]
    val_ids = [line.strip() for line in open('Flickr8k_text/Flickr_8k.devImages.txt').readlines()]
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
    data_gen = DataGenerator(train_ids, annotations, len(tokenizer.word_index) + 1, tokenizer, max_length, batch_size=5)

    # embedding_matrix = get_embedding(tokenizer)
    import pickle
    embedding_matrix = pickle.load(open('embedding.pkl', 'rb'))
    model = Concat_LSTM_GloVe(vocab_size=len(tokenizer.word_index) + 1,
                              embedding_dim=300,
                              embedding_matrix= embedding_matrix,
                              max_length=max_length, 
                              dropout=0.5)
    # model.load_weights('models/model256_LSTM_inject/weights.80-0.52.hdf5')

    path = 'models/modelGlove_LSTM256_inject'
    import os 
    os.makedirs(path, exist_ok=True)
    batch_size = 20

    checkpoint = ModelCheckpoint(path + "/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    csvlog = CSVLogger(path+'_train_log.csv',append=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5)

    generator_train = DataGenerator(train_ids, annotations, len(tokenizer.word_index) + 1, tokenizer, max_length, batch_size=batch_size)
    generator_val = DataGenerator(val_ids, annotations, len(tokenizer.word_index) + 1, tokenizer, max_length, batch_size=batch_size, shuffle=False)

    model.fit(generator_train, steps_per_epoch=len(generator_train),
            validation_data=generator_val, validation_steps=len(generator_val),
            epochs = 50, verbose = 1, initial_epoch = 0,
            callbacks = [checkpoint, csvlog, early_stopping])
