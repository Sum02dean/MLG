import os
from collections import Counter

import numpy as np
import pyfasta

GENOME_FILE = f'../data/hg38.fa'
assert os.path.exists(GENOME_FILE)
GENOME = pyfasta.Fasta(GENOME_FILE)

SEQ_TO_ONEHOT = {
    'N': [0, 0, 0, 0],
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1]
}


def get_sequence(chromosome_nr: int, start: int, stop: int) -> str:
    """
    Returns sequence as string in given chromosome with positions [start, stop].

    :param chromosome_nr: chromosome from which to read sequence
    :param start: first nucleotide position to read
    :param stop: last nucleotide position to read (includes stop!)
    :return: sequence as a string
    """
    return GENOME.sequence({'chr': f'chr{chromosome_nr}', 'start': start, 'stop': stop})


def clean_seq(seq: str) -> str:
    """
    Cleans the sequence: every letter is in 'ATCGN'.

    :param seq: Input sequence
    """
    seq = seq.upper()

    assert lambda key: key in ['A', 'T', 'C', 'G', 'N'], Counter(seq).keys()
    return seq


def encode_seq(seq: str) -> np.ndarray:
    """
    Converts given input sequence into a one-hot encoded array.
    'N' is represented as all zeros.

    :param seq: Input sequence.
    :return: One-hot encoded sequence as numpy array.
    """
    seq = clean_seq(seq)
    return np.array([SEQ_TO_ONEHOT[nucleotide] for nucleotide in seq])


def get_encoded_sequence(chromosome_nr: int, start: int, stop: int) -> np.ndarray:
    """
    Returns one-hot encoded sequence in given chromosome with positions [start, stop].

    :param chromosome_nr: chromosome from which to read sequence
    :param start: first nucleotide position to read
    :param stop: last nucleotide position to read (includes stop!)
    :return: One-hot encoded sequence as numpy array
    """
    return encode_seq(get_sequence(chromosome_nr, start, stop))


if __name__ == '__main__':
    sequence = get_sequence(1, 10000, 20000)
    print(sequence)
    print(encode_seq(sequence))
