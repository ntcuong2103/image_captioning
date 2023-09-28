import os

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Embedding,
    Input,
    RepeatVector,
    Reshape,
    concatenate,
)
from tensorflow.keras.models import Model

def Injection_LSTM(
    input_size,
    hidden_size,
    vocab_size,
    embedding_dim,
    embedding_matrix,
    max_length,
    dropout,
):
    # feature extractor model
    inputs1 = Input(shape=(input_size,))
    fe1 = Dropout(dropout)(inputs1)
    fe2 = Dense(hidden_size, activation="relu")(fe1)
    fe3 = RepeatVector(max_length)(fe2)

    # sequence model
    inputs2 = Input(shape=(max_length,))
    if type(embedding_matrix) != np.ndarray:
        se1 = Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_length,
            mask_zero=True,
            trainable=True,
        )(inputs2)
    else:
        se1 = Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_length,
            weights=[embedding_matrix],
            mask_zero=True,
            trainable=False,
        )(inputs2)

    se2 = concatenate([fe3, se1])
    se3 = Dropout(dropout)(se2)
    se4 = LSTM(hidden_size, return_sequences=True)(se3)

    # decoder model
    decoder1 = Dense(hidden_size, activation="relu")(se4)
    outputs = Dense(vocab_size, activation="softmax")(decoder1)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    # summarize model
    print(model.summary())

    return model


def Merge_LSTM(
    input_size,
    hidden_size,
    vocab_size,
    embedding_dim,
    embedding_matrix,
    max_length,
    dropout,
):
    inputs1 = Input(shape=(input_size,))
    fe1 = Dropout(dropout)(inputs1)
    fe2 = Dense(hidden_size, activation="relu")(fe1)
    fe3 = RepeatVector(max_length)(fe2)

    # GRU sequence model
    inputs2 = Input(shape=(max_length,))
    if type(embedding_matrix) != np.ndarray:
        se1 = Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_length,
            mask_zero=True,
            trainable=True,
        )(inputs2)
    else:
        se1 = Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_length,
            weights=[embedding_matrix],
            mask_zero=True,
            trainable=False,
        )(inputs2)
    se2 = Dropout(dropout)(se1)
    se3 = LSTM(hidden_size, return_sequences=True)(se2)

    # Merging both models
    decoder1 = concatenate([fe3, se3])
    decoder2 = Dropout(dropout)(decoder1)
    decoder3 = Dense(hidden_size, activation="relu")(decoder2)
    outputs = Dense(vocab_size, activation="softmax")(decoder3)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def Attention_LSTM(input_size, hidden_size, vocab_size, embedding_dim, embedding_matrix, max_length, dropout):
    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(
        shape=(
            None,
            None,
            input_size,
        )
    )
    feat_seq = Reshape((-1, input_size))(inputs1)
    fe1 = Dropout(dropout)(feat_seq)
    fe2 = Dense(hidden_size, activation="relu")(fe1)

    inputs2 = Input(shape=(max_length,))

    if type(embedding_matrix) != np.ndarray:
        se1 = Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_length,
            mask_zero=True,
            trainable=True,
        )(inputs2)
    else:
        se1 = Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_length,
            weights=[embedding_matrix],
            mask_zero=True,
            trainable=False,
        )(inputs2)


    # initial state from avg of image features
    avg_features = tf.keras.layers.GlobalAveragePooling2D()(inputs1)

    initial_hidden_state = Dense(hidden_size, activation="relu")(avg_features)
    initial_cell_state = Dense(hidden_size, activation="relu")(avg_features)

    initial_state = [initial_hidden_state, initial_cell_state]

    se2 = Dropout(dropout)(se1)
    se3 = LSTM(hidden_size, return_sequences=True)(se2, initial_state = initial_state)

    attention_context = tf.keras.layers.AdditiveAttention()([se3, fe2])

    # Merging both models
    decoder1 = concatenate([attention_context, se3])
    decoder2 = Dropout(dropout)(decoder1)
    decoder3 = Dense(hidden_size, activation="relu")(decoder2)
    outputs = Dense(vocab_size, activation="softmax")(decoder3)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(
        loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"]
    )

    return model

