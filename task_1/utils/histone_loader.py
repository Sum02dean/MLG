import os.path

import pyBigWig

HISTONE_MODS: list[str] = ['DNase', 'H3K4me1', 'H3K4me3', 'H3K9me3', 'H3K27ac', 'H3K27me3', 'H3K36me3']
VALUE_TYPES = ['mean', 'max', 'min', 'coverage', 'std']


def get_histone_file(histone: str, cell_line: int) -> str:
    """
    Find the histone modification data for a given histone modification and cell line with the correct bigwig extension.

    :param histone: histone file name (directory name where exists bigwig file for all cell lines)
    :param cell_line: cell line
    :return: path to histone modification bigwig file
    """
    path = f'../data/{histone}-bigwig/X{cell_line}'
    file_extensions = ['.bw', '.bigwig']
    return list(filter(lambda p: os.path.exists(p), [path + ext for ext in file_extensions]))[0]


def get_bw_data(cell_line: int, chr: int, start: int, stop: int, value_type: str = 'mean', histones=None):
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
    if histones is None:
        histones = HISTONE_MODS
    assert cell_line in [1, 2, 3]
    assert chr in range(1, 23)
    assert value_type in VALUE_TYPES
    assert all(histone in HISTONE_MODS for histone in histones)

    stats = []
    for histone in histones:
        filename = get_histone_file(histone, cell_line)
        bw = pyBigWig.open(filename)
        stat = bw.stats(f'chr{chr}', start, stop, type=value_type)
        # TODO: there is also an option to get a list of values for multiple bins!
        bw.close()
        stats += stat

    return stats


if __name__ == '__main__':
    for cell_line in [1, 2, 3]:
        print(get_bw_data(cell_line, 1, 10000, 10100))
