from models.baselines import set_seed, random_forest
from utils.histone_loader import HISTONE_MODS


def analyse_histone_contribution():
    """
    RF performance
    DNase:      0.6885
    H3K4me1:    0.5321
    H3K4me3:    0.7126
    H3K9me3:    0.3205
    H3K27ac:    0.6910
    H3K27me3:   0.4446
    H3K36me3:   0.3826

    ALL: 0.7511
    without H3K9me3: 0.7498
    without H3K9me3 and H3K36me3: 0.7372

    ALL mean:   0.7511
    ALL max:    0.7484
    ALL min:    0.7511
    ALL cov.:   0.0077
    ALL std:    0.7481
    """
    set_seed()
    for histone in HISTONE_MODS:
        random_forest(histone_mods=[histone])


def analyse_window_and_bin():
    """
    RESULTS:
    bin     flank   score
    20      1000    0.7474
    50      1000    0.7510
    100     1000    0.7484
    200     1000    0.7474
    250     1000    0.7461
    500     1000    0.7424
    1000    1000    0.7296
    2000    1000    0.7200

    200     1000    0.7476
    200     2000    0.7372
    200     5000    0.7341
    200     10000   0.7351
    200     20000
    """
    set_seed()
    params = {
        'window_size': [5000],
        'bin_size': [200],
    }
    for flank in params['window_size']:
        for bin_size in params['bin_size']:
            random_forest(flank_size=flank, bin_size=bin_size)
