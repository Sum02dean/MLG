import os

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
COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


def clean_seq(seq: str) -> str:
    """
    Cleans the sequence: every letter is in 'ATCGN'.

    :param seq: Input sequence
    """
    seq = seq.upper()

    assert lambda char: char in ['A', 'T', 'C', 'G', 'N'], set(seq)
    return seq


def get_sequence(chromosome_nr: int, start: int, stop: int, is_reverse: bool) -> str:
    """
    Returns sequence as string in given chromosome with positions [start, stop].
    For reverse strand, returns the reverse complement.

    :param chromosome_nr: chromosome from which to read sequence
    :param start: first nucleotide position to read
    :param stop: last nucleotide position to read (includes stop!)
    :param is_reverse: must be set to True for reverse strand
    :return: sequence as a string
    """
    seq = GENOME.sequence({'chr': f'chr{chromosome_nr}', 'start': start, 'stop': stop})
    seq = clean_seq(seq)
    if is_reverse:
        return ''.join(COMPLEMENT[base] for base in reversed(seq))
    return seq


def encode_seq(seq: str) -> list:
    """
    Converts given input sequence into a one-hot encoded array.
    'N' is represented as all zeros.

    :param seq: Input sequence.
    :return: One-hot encoded sequence.
    """
    return [SEQ_TO_ONEHOT[nucleotide] for nucleotide in seq]


def get_encoded_sequence(chromosome_nr: int, start: int, stop: int, is_reverse: bool) -> list:
    """
    Returns one-hot encoded sequence in given chromosome with positions [start, stop].

    :param chromosome_nr: chromosome from which to read sequence
    :param start: first nucleotide position to read
    :param stop: last nucleotide position to read (includes stop!)
    :param is_reverse: must be set to True for reverse strand
    :return: One-hot encoded sequence
    """
    return encode_seq(get_sequence(chromosome_nr, start, stop, is_reverse))


if __name__ == '__main__':
    sequence = get_sequence(1, 10000, 10010, False)
    print(sequence)
    print(encode_seq(sequence))
    sequence = get_sequence(1, 10000, 10010, True)
    print(sequence)
    print(encode_seq(sequence))
