from data import get_tokenizer, DataGenerator
from model import define_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

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

    model = define_model(len(tokenizer.word_index) + 1, 512, max_length, 0.5)
    model.load_weights('models/model256_LSTM_inject/weights.19-0.43.hdf5')

    path = 'models/model256_LSTM_inject'
    import os 
    os.makedirs(path, exist_ok=True)
    batch_size = 5

    checkpoint = ModelCheckpoint(path + "/weights.{epoch:02d}-{val_masked_accuracy:.2f}.hdf5", monitor='val_masked_accuracy', verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
    csvlog = CSVLogger(path+'_train_log.csv',append=True)
    early_stopping = EarlyStopping(monitor='val_masked_accuracy', min_delta=0, patience=5)

    generator_train = DataGenerator(train_ids, annotations, len(tokenizer.word_index) + 1, tokenizer, max_length, batch_size=batch_size)
    generator_val = DataGenerator(train_ids, annotations, len(tokenizer.word_index) + 1, tokenizer, max_length, batch_size=batch_size)

    model.fit(generator_train, steps_per_epoch=len(generator_train),
            validation_data=generator_val, validation_steps=len(generator_val),
            epochs = 30, verbose = 1, initial_epoch = 19,
            callbacks = [checkpoint, csvlog, early_stopping])
