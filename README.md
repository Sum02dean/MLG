# MLG
Machine Learning in Genomics Course ETH.

# THE AMOEBAS

![APM](https://img.shields.io/apm/l/vim-mode) 
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg?style=flat)](https://github.com/feross/standard)
<img src="amoeba.png" width="450">

## Moodle: 
https://moodle-app2.let.ethz.ch/course/view.php?id=16540

## Project server:
https://project.ml4g.ethz.ch/


# Project 1 - Prediction of Gene Expression from Chromatin Landscape

## Setup
Dependencies for conda and pip are listed in `environment.yml` and `requirements.txt`.
In the project base folder execute in the command line:
```commandline
conda env create -f environment.yml
pip install -r requirements.txt
```

## Data
The [project data](https://polybox.ethz.ch/index.php/s/iY6d8qbMMiy4dQh) should be unzipped into `/task_1/data`. For example, the X1 dataset train info should be available at `/task_1/data/CAGE-train/X1_train_info.tsv`.

We're using the human reference genome version hg38. It should be downloaded into the folder `data` from `https://s3.amazonaws.com/igv.broadinstitute.org/genomes/seq/hg38/hg38.fa`.

## Dependencies
- Histone modification data processing with [pyBigWig](https://github.com/deeptools/pyBigWig).
