import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Embedding, Dropout, RepeatVector, add, concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tqdm import tqdm
import os
from tensorflow.keras.preprocessing.text import Tokenizer

def extract_features(directory):
    '''
    input_shape: optional shape tuple, only to be specified
      if `include_top` is `False` (otherwise the input shape
      has to be `(299, 299, 3)` (with `'channels_last'` data format)
      or `(3, 299, 299)` (with `'channels_first'` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 75.
    '''
    model = InceptionResNetV2(include_top = False, weights='imagenet', pooling='avg')
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img
        image = load_img(filename)
        image = img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        # print (image.shape)
        image = preprocess_input(image)
            
        feature = model.predict(image)
        features[img] = feature
    return features

def masked_categorical_crossentropy(y_true, y_pred):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction="none",
    )
    mask = tf.cast((tf.argmax(y_true, -1) != 0), dtype=tf.float32)
    loss = tf.reduce_sum(loss_fn (y_true, y_pred) * mask, -1) / tf.reduce_sum(mask, -1)
    return tf.reduce_mean(loss)

def masked_accuracy(y_true, y_pred):
    mask = tf.cast((tf.argmax(y_true, -1) != 0), dtype=tf.float32)
    accuracy = tf.reduce_sum(tf.cast(tf.argmax(y_true, -1) == tf.argmax(y_pred, -1), dtype=tf.float32) * mask, -1) / tf.reduce_sum(mask, -1)
    return tf.reduce_mean(accuracy)

def Concat_LSTM(vocab_size, embedding_dim, max_length, dropout):
    # feature extractor model
    inputs1 = Input(shape=(1536,))
    fe1 = Dropout(dropout)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)
    fe3 = RepeatVector(max_length)(fe2)
    
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True, trainable=True)(inputs2)

    se2 = concatenate([fe3, se1])
    se3 = Dropout(dropout)(se2)
    se4 = LSTM(512, return_sequences=True) (se3)
    
    # decoder model
    decoder1 = Dense(512, activation='relu')(se4)
    outputs = Dense(vocab_size, activation='softmax')(decoder1)
    
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # model.compile(loss=[masked_categorical_crossentropy], optimizer='adam', metrics=[masked_accuracy])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize model
    print(model.summary())
     
    return model

def Concat_LSTM_GloVe(vocab_size, embedding_dim, max_length, embedding_matrix, dropout):
    # feature extractor model
    inputs1 = Input(shape=(1536,))
    fe1 = Dropout(dropout)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)
    fe3 = RepeatVector(max_length)(fe2)
    
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], mask_zero=True, trainable=True)(inputs2)

    se2 = concatenate([fe3, se1])
    se3 = Dropout(dropout)(se2)
    se4 = LSTM(512, return_sequences=True) (se3)
    
    # decoder model
    decoder1 = Dense(512, activation='relu')(se4)
    outputs = Dense(vocab_size, activation='softmax')(decoder1)
    
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # model.compile(loss=[masked_categorical_crossentropy], optimizer='adam', metrics=[masked_accuracy])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize model
    print(model.summary())
     
    return model
if __name__ == "__main__":
    # import numpy as np
    # vocab_size = 10000
    # max_length = 256
    # embedding_dim = 300
    # dropout = 0.25
    # embedding_matrix = np.random.rand(vocab_size, embedding_dim)
    # define_model(vocab_size, max_length, embedding_dim, dropout, embedding_matrix)

    # features = extract_features('Images')
    # import pickle
    # pickle.dump(features, open('img_features.pkl', 'wb'))
    # exit()


    # model = load_model('saved_models/model256_GRU_inject_dropout0.25.h5')
    # # load and prepare the photograph
    # # image = "C:\\Users\\HDH\\Desktop\\python-img-cap-gen\\Flicker8k_Dataset\\3484649669_7bfe62080b.jpg"
    # image = "C:\\Users\\HDH\\Desktop\\python-img-cap-gen\\Flicker8k_Dataset\\3484649669_7bfe62080b.jpg"
    # photo = extract_features(image)
    # # generate description
    # description = generate_desc(model, tokenizer, photo, max_length)
    # print(description)
    pass
