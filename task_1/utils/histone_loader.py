import os
from enum import Enum, auto

import pandas as pd
import pyBigWig
import pybedtools


class HistoneMod(Enum):
    DNase = auto()
    H3K4me1 = auto()
    H3K4me3 = auto()
    H3K9me3 = auto()
    H3K27ac = auto()
    H3K27me3 = auto()
    H3K36me3 = auto()


BED_NARROW_PEAK_COLUMNS = ['chr', 'start', 'end', 'name', 'score', 'strand', 'signal_value', 'p_value', 'q_value', 'peak']
# NB! end is not part of interval
HISTONE_MODS = list(HistoneMod.__members__.keys())
BED_MODS: dict[HistoneMod, list[str]] = {histone: ['X1.bed', 'X2.bed', 'X3.bed'] for histone in HistoneMod}
BW_MODS: dict[HistoneMod, list[str]] = {
    HistoneMod.DNase: ['X1.bw', 'X2.bw', 'X3.bigwig'],
    HistoneMod.H3K27ac: ['X1.bigwig', 'X2.bw', 'X3.bw']
}
VALUE_TYPES = ['mean', 'max', 'min', 'coverage', 'std']


def load_bed_df(histone_mod: HistoneMod, filename: str):
    return pd.read_csv(f'..{os.sep}data{os.sep}{histone_mod.name}-bed{os.sep}{filename}', sep='\t', header=None,
                       names=BED_NARROW_PEAK_COLUMNS)


def get_bed_data(cell_line: int, chr: int, start: int, stop: int, value_type: str = 'mean',
                 histones: list[HistoneMod] = BED_MODS.keys()):
    """

    NB! end in bed line is not included in interval
    NB! assumes no overlapping intervals

    :param cell_line:
    :param chr:
    :param start:
    :param stop:
    :param value_type:
    :param histones:
    """
    assert cell_line in [1, 2, 3]
    assert chr in range(1, 23)
    assert value_type in VALUE_TYPES
    assert all(histone in BED_MODS for histone in histones)

    stats = []
    for histone in histones:
        filename = BED_MODS[histone][cell_line - 1]
        bed_df = load_bed_df(histone, filename)
        bed_df = bed_df[bed_df.chr == f'chr{chr}']
        df = bed_df.sort_values('start')
        prev_end = 0
        for index, line in df.iterrows():
            assert line.start >= prev_end
            prev_end = line.end
        bed_df = bed_df[bed_df.end > start]
        bed_df = bed_df[bed_df.start <= stop]

        print(bed_df.head())

    pass


def get_bw_data(cell_line: int, chr: int, start: int, stop: int, value_type: str = 'mean',
                histones: list[HistoneMod] = BW_MODS.keys()):
    """
    Get values from given histone marks for bigwig files.

    :param cell_line: cell line in [1, 2, 3]
    :param chr: chromosome number in [1..22]
    :param start: start index in given chromosome
    :param stop: stop index in given chromosome (included in stats!)
    :param value_type: value type over given interval (in ['mean', 'max', 'min', 'coverage', 'std'])
    :param histones: list of bigwig histone modification to calculate
    :return: averaged (by value type) value for given histone modifications from bigwig files
    """
    assert cell_line in [1, 2, 3]
    assert chr in range(1, 23)
    assert value_type in VALUE_TYPES
    assert all(histone in BW_MODS for histone in histones)

    stats = []
    for histone in histones:
        filename = BW_MODS[histone][cell_line - 1]
        bw = pyBigWig.open(f'../data/{histone.name}-bigwig/{filename}')
        stat = bw.stats(f'chr{chr}', start, stop, type=value_type)
        # TODO: there is also an option to get a list of values for multiple bins!
        bw.close()
        stats += stat

    return stats


if __name__ == '__main__':
    print(get_bw_data(1, 1, 10000, 10100))
    # print(get_bed_data(1, 1, 10000, 10100))
