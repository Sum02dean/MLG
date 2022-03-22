from enum import Enum, auto

import pyBigWig


class HistoneMod(Enum):
    DNase = auto()
    H3K4me1 = auto()
    H3K4me3 = auto()
    H3K9me3 = auto()
    H3K27ac = auto()
    H3K27me3 = auto()
    H3K36me3 = auto()


HISTONE_MODS = list(HistoneMod.__members__.keys())
BED_MODS = HISTONE_MODS
BW_MODS: dict[HistoneMod, list[str]] = {HistoneMod.DNase: ['X1.bw', 'X2.bw', 'X3.bigwig'],
                                        HistoneMod.H3K27ac: ['X1.bigwig', 'X2.bw', 'X3.bw']}
VALUE_TYPES = ['mean', 'max', 'min', 'coverage', 'std']


def get_bw_data(cell_line: int, chr: int, start: int, stop: int, value_type: str = 'mean',
                histones: list[HistoneMod] = BW_MODS.keys()):
    assert value_type in VALUE_TYPES

    stats = []
    for histone in histones:
        filename = BW_MODS[histone][cell_line]
        bw = pyBigWig.open(f'../data/{histone.name}-bigwig/{filename}')
        stat = bw.stats(f'chr{chr}', start, stop, type=value_type)
        # TODO: there is also an option to get a list of values for multiple bins!
        stats += stat

    return stats


if __name__ == '__main__':
    print(get_bw_data(1, 1, 10000, 10100))
