Welcome to the MLG wiki! Here we can share some notes and ideas to discuss.

## Task:
Predict gene expression from epigenetic code and sequence data.

## Biology:
Epigenteics give insights into the transcriptional activity of genes. Common "data-tracks" are listed:

* DNAse: ChIP-seq data - pulls down regions on open-chromatin (unwound DNA) i.e. areas where no histones proteins are present.

[//]: # not sure how DNAse and CHIP-seq data is related
* Epigenetic marker tracks - Show transcription factor binding. The correlation between combinatorial TFs and gene activation/repression is not obvious.

[//]: #  From my understanding, epigenetic marker tracks are [CHIP-seq data show histone modification](https://ccg.epfl.ch/chipseq/doc/chipseq_tutorial_intro.php). It might correlated with TF binding but not directly provide such information ?

[//]: # _**"not sure how DNAse and CHIP-seq data is related..."**_ - good catch! They are two different methods.

* Sequence annotation - Information about 5'-3' untranslated regions (UTRs), transcription start sties(TSS), intronic (non-coding) and exonic (coding) DNA regions. 
* Nucleotide sequence 


## File types:
Some common file types
* [.bed](https://genome.ucsc.edu/FAQ/FAQformat.html#format12) - Used to store genomic regions as coordinates and associated annotations.
* [.bigwig](https://software.broadinstitute.org/software/igv/bigwig) - Used to store dense, continuous data that will be displayed in the Genome Browser as a graph.
* [.fa](https://software.broadinstitute.org/software/igv/FASTA) - Fasta format, used to specify the reference genome sequence.


## Pre-processing and data:
* Features - epigenetic data and/or sequene data [edited] 
* Labels - Pseudo-log2 normalized gene expression data [edited]

[//]: #  i think both pseudo-log2 normalised count data  and raw count data refer to gene expression that we need to predict 

[//]: # **_"i think both pseudo-log2 normalised count ..."_** - Right again! 

## Modeling task (regression, classification, NLP, etc):
Regression

## Model(s):
Ideally we create a simple model first... nothing fancy. Model should at least be non-linear due to high dimensional data,
and the combinatorics/analog/redundant nature of protein-DNA binding modes. - Can also try a simple linear model to benchmark. 

Initial model ideas:
* RBF kernal SVM / benchmark with linear RBF
* Basic neural network with one or two layers with non-linearity 
* Gaussian process with quadratic kernel or mixed kernels

## Misc:
* Intuitively - We are more interested in regions containing and flanking TSS (1-4 Mbp). Intronic regions are interesting **IF** high density DNAse signal present. Note that the deep learning framework [ExPecto](https://www.nature.com/articles/s41588-018-0160-6) used DNA regions in the order of 40kbp for promoter proximal regions. 


* We could make use of the sequence either explicitly (e.g. DNA position present in column order for tabular data) or explicit - use nucleotide sequence (k-mers) as a feature. Else we use non-spatial oriented design matrix.

* Since epigenetic marks work in combination and can be either active or repressive to variable extents, we should simply provide their normalized density as features for carefully chosen marks instead of defining categorically present/absent OHE.

* Identification of epigenetic marks can be chosen by examining the most highly variable sets for a given cell line (or across cell-lines) for given genes. This is similar to the idea in highly variable gene analysis for ScRNA-seq genomics. High variability across differential expressed genes could signify a valid biological association. Once we select for the top-n variable marks, we can draw up a correlation table with the labels and see what signal is present in the context of gene expression (neg-corr, pos-corr, zero-corr). 

* Perhaps could use spatial auto-correlation e.g. [Moran's I](https://www.youtube.com/watch?v=OJU8GNW9grc) to identify associations based on mixture of marker signal and genomic sequence position. 

* We should probably rely heavily on  DNAse as a heuristic to prioritize sequence locality and markers that co-localize at these points.





## Resources/Papers:
| Year | Paper                                                                                                       | Input                                                                                                                                                                                                                                     | Method                                                             |
|------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| 2021 | [Prediction gene expression levels...](https://www.sciencedirect.com/science/article/pii/S0169743921002240) | 11 histone marks (H3K36me3, H3K9ac, H3K27me3, H3K4me3, H3K27ac, H3K4me2, H3K79me2, H3K9me3, H4K20me1, H3K4me1, H2A.Z) from TSS (+-5000bp) and TTS(+-2000bp) as feature maps. bigWigSummary for median histone signal for each bin (100bp) | block of conv + LSTM for both TSS and TTS, concat and FC           |
| 2018 | [Expecto](https://www.nature.com/articles/s41588-018-0160-6)                                                | DNA sequence                                                                                                                                                                                                                              | CNN for pred chromatin expression, then regularized linear models  |
| 2016 | [DeepChrome](https://academic.oup.com/bioinformatics/article/32/17/i639/2450757)                            | histone marks                                                                                                                                                                                                                             | CNN                                                                |
| 2019 | [DeepHistone](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-5489-4)                     | seq + 7 histone marks (H3K4me3, H3K4me1, H3K36me3, H3K27me3, H3K9me3, H3K27ac, and H3K9ac)                                                                                                                                                | CNN for both DNA and DNase module                                  |
