
import torch
import numpy as np

'''
Class implementing a sampler for inference on a model. Given the raw logits from
an LLM model, this will sample the next token id.
'''
class Sampler:

    def __init__(
        self,
        top_k=None,
        top_p=None,
        frequency_penalty=1.0,
        presence_penalty=1.0
    ):
        '''
        param top_k : (None or int)
            If specified, only the top k logits should be used during sampling
            If this is specified, top_p should be None

        param top_p : (None or int)
            If specified, only the logits representing the probability mass p should be used during sampling.
            Or, if the top token has mass greater than p, the top token is returned.
            If this is specified, top_k should be None

        If top_k and top_p are both None, sample from the whole distribution (same as top_p=1.0)

        param frequency_penalty : (float)
            A penalty applied to tokens that have previously occured in the sequence. Along with
            presence_penalty, this adjusts the per-token softmax temperature.
            A penalty of 1.0 indicates no change from normal softmax.

        param presence_penalty : (float)
            A penalty applied to tokens IF they have previously occured in the sequence. Along with
            frequency_penalty, this adjusts the per-token softmax temperature.
            A penalty of 1.0 indicates no change from normal softmax.
        '''
        if top_k is not None and top_p is not None:
            raise ValueError("top_k and top_p are mutually exclusive; set only one of them.")

        if top_k is not None:
            if not isinstance(top_k, (int, np.integer)) or top_k <= 0:
                raise ValueError("top_k must be a positive integer.")

        if top_p is not None:
            if not (0.0 < float(top_p) <= 1.0):
                raise ValueError("top_p must be in the range (0, 1].")

        if frequency_penalty < 1.0 or presence_penalty < 1.0:
            raise ValueError("frequency_penalty and presence_penalty should be >= 1.0.")

        self.top_k = top_k
        self.top_p = top_p
        self.frequency_penalty = float(frequency_penalty)
        self.presence_penalty = float(presence_penalty)


    def make_token_distribution(self, raw_unsorted_logits, previous_token_ids):
        '''
        param: raw_unsorted_logits (float numpy array)
            A one dimensional list of logits representing an unnormalized distribution over next tokens
            These are "unsorted" in the sense that their order aligns with vocabulary order, not with probability.

        param: previous_token_ids (int numpy array)
            A one dimensional list of ids representing the previous tokens, for calculating repetition penalties.

        returns:
            - the final probability distribution that this token is sampled from
            It should be returned back to token-id order (unsorted order) before returning.
        '''

        logits = np.asarray(raw_unsorted_logits, dtype=np.float64).copy()
        previous_token_ids = np.asarray(previous_token_ids, dtype=np.int64)

        if logits.ndim != 1:
            raise ValueError("raw_unsorted_logits must be 1D.")

        vocab_size = len(logits)

        # make temperature=1.0 for each vocabulary option
        temps = np.ones(vocab_size, dtype=np.float64)

        # adjust temps as needed with penalties
        if previous_token_ids.size > 0:
            valid_prev = previous_token_ids[(previous_token_ids >= 0) & (previous_token_ids < vocab_size)]
            if valid_prev.size > 0:
                counts = np.bincount(valid_prev, minlength=vocab_size).astype(np.float64)
                temps += counts * (self.frequency_penalty - 1.0)
                temps += (counts > 0).astype(np.float64) * (self.presence_penalty - 1.0)

        # logits = logits - np.min(logits) to make sure all are positive
        logits = logits - np.min(logits)

        # apply temps & softmax
        adjusted_logits = logits / temps
        adjusted_logits = adjusted_logits - np.max(adjusted_logits)
        dist = np.exp(adjusted_logits)
        dist = dist / np.sum(dist)

        # sort the distribution (and track the sort order so you can undo it later)
        indices = np.argsort(dist)[::-1]
        sorted_dist = dist[indices]

        # find either the top-p or top-k cutoff
        if self.top_k is not None:
            cutoff = min(self.top_k, vocab_size)
            kept = sorted_dist[:cutoff]
            kept_indices = indices[:cutoff]
        elif self.top_p is not None:
            cumulative = np.cumsum(sorted_dist)
            cutoff = np.searchsorted(cumulative, self.top_p) + 1
            kept = sorted_dist[:cutoff]
            kept_indices = indices[:cutoff]
        else:
            kept = sorted_dist
            kept_indices = indices

        # renormalize this portion by simply dividing by the sum
        kept = kept / np.sum(kept)

        # revert back to original ordering of the distribution
        final_dist = np.zeros_like(dist)
        final_dist[kept_indices] = kept

        # return distribution
        return final_dist



    #==========================
    # for actually sampling the distribution
    def sample_one_token(self, raw_unsorted_logits, previous_token_ids):
        probs = self.make_token_distribution(raw_unsorted_logits, previous_token_ids)
        return np.random.choice(np.arange(len(raw_unsorted_logits)), p=probs)

    # for convenience, this is also callable
    def __call__(self, raw_unsorted_logits, previous_token_ids):
        return self.sample_one_token(raw_unsorted_logits, previous_token_ids)




if __name__ == "__main__":
    
    # example of using this with dummy data, keeping everything in token ids

    sampler = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)

    sequence = [1,2,3,4,5]

    for i in range(10):
        # fake logits for a vocab of size 500
        logits = np.random.randn(500)

        # get next token in sequence
        next_token = sampler(logits, sequence)
        sequence.append(next_token)

    print(sequence)