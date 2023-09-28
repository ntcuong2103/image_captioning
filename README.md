# Image captioning: merge/inject/attention architecture

Reference: 
https://arxiv.org/pdf/1708.02043.pdf
https://arxiv.org/pdf/1703.09137.pdf
https://machinelearningmastery.com/caption-generation-inject-merge-architectures-encoder-decoder-model/

## Get image features (using encoder)
Check out [encoder.py](encoder.py)

## Create tokenizer, filter vocab
Check out [data.py](data.py)

Set MIN_WORD_COUNT to filter vocab with low freq.

## Get embedded matrix from pretrained embedding
Check out [embedding.py](embedding.py) 

## Train
Check out [train.py](train.py)

## Test
Check out [test.py](test.py)