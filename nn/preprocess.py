# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    pos_indices = np.where(np.array(labels))[0]
    neg_indices = np.where(~np.array(labels))[0]
    if len(neg_indices) > len(pos_indices):
        sampled_indices = [i for i in neg_indices]
        sampled_indices.extend(np.random.choice(pos_indices, len(neg_indices), 
                                                replace=True))
    else:
        sampled_indices = [i for i in pos_indices]
        sampled_indices.extend(np.random.choice(neg_indices, len(pos_indices), 
                                                replace=True))
    sampled_seqs = [seqs[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]
    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    return [one_hot_encode_seq(seq) for seq in seq_arr]


def one_hot_encode_seq(seq: str) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a single DNA sequence
    for use as input into a neural network.
    """
    encoding = []
    one_hot_dict = {'A': [1, 0, 0, 0], 
                    'T': [0, 1, 0, 0],
                    'C': [0, 0, 1, 0],
                    'G': [0, 0, 0, 1]}
    for aa in seq:
        encoding.extend(one_hot_dict[aa])
    return encoding
