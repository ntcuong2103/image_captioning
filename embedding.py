import numpy as np

def get_embedding(tokenizer):
    embeddings_index = {} 
    f = open('models/modelGlove_LSTM256_inject/glove.42B.300d.txt', encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    embedding_dim = 300
    hits = 0
    misses = 0
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits +=1
        else:
            misses +=1
    print("Converted %d words (%d misses)" % (hits, misses))
    import pickle
    pickle.dump(embedding_matrix, open('embedding.pkl', 'wb'))
    return embedding_matrix

if __name__ == '__main__':
    from data import get_tokenizer
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
    get_embedding(tokenizer)
