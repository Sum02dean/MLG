import os

import pyfasta


GENOME_FILE = f'../data/hg38.fa'
assert os.path.exists(GENOME_FILE)

GENOME = pyfasta.Fasta(GENOME_FILE)


def get_sequence(chromosome_nr: int, start: int, stop: int) -> str:
    return GENOME.sequence({'chr': f'chr{chromosome_nr}', 'start': start, 'stop': stop})


if __name__ == '__main__':
    print(get_sequence(1, 181415, 181565))
