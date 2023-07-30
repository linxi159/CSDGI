# CSDGI
Unravelling Cancer Subtype-specific Driver Genes in Single-cell Transcriptomics Data with CSDGI
![](https://github.com/linxi159/AAFL/blob/main/figures/CSDGI.jpg) 

## Description of each directory
data: the preprocessed data from real scRNA-seq data in GEO.

comparison: the utility of comparison with different methods.

results: the preprocessed results and final results.

figures: the plot for CSDGI.


## How to setup

* Python (3.6 or later)

* numpy

* sklearn

* pytorch

* NVIDIA GPU + CUDA 11.50 + CuDNN v7.1

* scipy


## Quick example to use AAFL
```
* train and test the model:

* the implementation of driver gene identification between normal cells and tumor cells
python main1_driver_gene_inferring_normal_tumor.py

* the implementation of cancer subtype-specific driver gene identification across potential cancel subtypes
python main2_subtype_driver_gene_inferring_tumor.py

```

