import heapq

import numpy as np


def generate_sequence_beamsearch(
    predictions_function, tokenizer, beam_width=3, clip_len=20
):
    start_index = tokenizer.word_index["<s>"]
    end_index = tokenizer.word_index["<e>"]
    unknown_index = tokenizer.word_index["<unk>"]
    prev_beam = Beam(beam_width)
    prev_beam.add(np.array(1.0, "float64"), False, [start_index])
    while True:
        curr_beam = Beam(beam_width)

        # Add complete sentences that do not yet have the best probability to the current beam, the rest prepare to add more words to them.
        prefix_batch = list()
        prob_batch = list()
        for prefix_prob, complete, prefix in prev_beam:
            if complete == True:
                curr_beam.add(prefix_prob, True, prefix)
            else:
                prefix_batch.append(prefix)
                prob_batch.append(prefix_prob)

        # Get probability of each possible next word for each incomplete prefix.
        indexes_distributions = predictions_function(prefix_batch)

        # Add next words
        for prefix_prob, prefix, indexes_distribution in zip(
            prob_batch, prefix_batch, indexes_distributions
        ):
            for next_index, next_prob in enumerate(indexes_distribution):
                if next_index == unknown_index:  # skip unknown tokens
                    pass
                elif (
                    next_index == end_index
                ):  # if next word is the end token then mark prefix as complete and leave out the end token
                    curr_beam.add(prefix_prob * next_prob, True, prefix)
                else:  # if next word is a non-end token then mark prefix as incomplete
                    curr_beam.add(
                        prefix_prob * next_prob, False, prefix + [next_index]
                    )

        (best_prob, best_complete, best_prefix) = max(curr_beam)
        if (
            best_complete == True or len(best_prefix) - 1 == clip_len
        ):  # if the length of the most probable prefix exceeds the clip length (ignoring the start token) then return it as is
            return " ".join(
                tokenizer.index_word[index] for index in best_prefix[1:]
            )  # return best sentence without the start token

        prev_beam = curr_beam


class Beam(object):
    # For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
    # This is so that if two prefixes have equal probabilities then a complete sentence is preferred over an incomplete one since (0.5, False) < (0.5, True)

    #################################################################
    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    #################################################################
    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    #################################################################
    def __iter__(self):
        return iter(self.heap)
