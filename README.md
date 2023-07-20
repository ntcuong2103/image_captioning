# Image captioning: merge architecture

Reference: 
https://arxiv.org/pdf/1708.02043.pdf
https://arxiv.org/pdf/1703.09137.pdf
https://machinelearningmastery.com/caption-generation-inject-merge-architectures-encoder-decoder-model/

# Result
## Trained embedding, concatenate features
BLEU-1: 0.561931
BLEU-2: 0.376341
BLEU-3: 0.243623
BLEU-4: 0.154181
## GloVe embedding, concatenate features
BLEU-1: 0.566866
BLEU-2: 0.376281
BLEU-3: 0.242561
BLEU-4: 0.153092
## Trained embedding, merge features (add)
BLEU-1: 0.524299
BLEU-2: 0.352539
BLEU-3: 0.229101
BLEU-4: 0.144337
## Attention models (Inception ResNet)
BLEU-1: 0.540193
BLEU-2: 0.355168
BLEU-3: 0.220169
BLEU-4: 0.133090
