from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
import os
import numpy as np


def InceptionResNetV2Encoder(pooling='avg'):
    model = InceptionResNetV2(
        include_top=False, weights="imagenet", pooling=pooling
    )
    if pooling == 'none':
        model = Model(
            inputs=model.input, outputs=model.get_layer("conv_7b").output
        )
    return model

def ResNet50Encoder(pooling='avg'):
    model = ResNet50(include_top=False, weights="imagenet", pooling=pooling)
    if pooling == 'none':
        model = Model(inputs=model.input, outputs = model.get_layer('conv5_block3_out').output)
    return model

def extract_features(directory, model):
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img
        image = load_img(filename)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        feature = model.predict(image)
        features[img] = feature
    return features

if __name__ == "__main__":
    model = InceptionResNetV2Encoder(pooling='avg')
    features = extract_features('Images', model)
    import pickle
    pickle.dump(features, open('models/encoded_features/inception_resnet_v2_avg.pkl', 'wb'))
    exit()
