# 2024-phenotypeomat

[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/projects/miniconda/en/latest/)

## Purpose

This repository contains the analysis code to create the data driven components of the figures in our pub [The phenotype-o-mat: A flexible tool for collecting visual phenotypes](LINK_TO_PUB)
These scripts are intended to provide exampls of how an experimentor might analyze data collected on the Phenotype-o-mat.
A comprehensive protocol for assembling a Phenotype-o-mat is available [here](https://www.protocols.io/view/building-a-phenotype-o-mat-a-low-cost-diy-plate-re-yxmvm3r3ol3p/v1)

## Installation and Setup

This repository uses conda to manage software environments and installations. You can find operating system-specific instructions for installing miniconda [here](https://docs.conda.io/projects/miniconda/en/latest/). After installing conda and [mamba](https://mamba.readthedocs.io/en/latest/), run the following command to create the pipeline run environment.

```{bash}
mamba env create -n phenotypeomat-analysis --file envs/dev.yml
conda activate phenotypeomat-analysis
```

After setting up the conda environment, the analyses can be run as follows:

`python3 colony_segment_figure_chr_fl_fig.py [PATH TO SINGLE C. REINHARDTII TIFF] [PATH TO SINGLE C. SMITHII TIFF]`

`python3 parent_strains_reflectance_fig.py [PATH TO FOLDER CONTAINING IMAGES]`

## Data

The data analyzed for the pub are available [here](PATH TO ZENODO LIB)

## Overview

### Description of the folder structure

The analysis scripts are located in the `data_analysis_scripts` folder.
The dev.yml file defines the conda envionrment to run the scripts and is contained in the `env` folder.

### Methods

These two scripts analyze and plot two types of data collected using the phenotypeomat: chlorophyll fluorescence data and multi-wavelength reflectance data.
Each script starts by taking a single image (flurescence or transillumination) and identifying colony location and shape. These segmentations are then used to collect the remaining intensity data (fluorescence or reflectance) and plot that data. The details of those plots can be seen in the pub or by running the scripts.

### Compute Specifications
The computer used to run these analyses:
CPU: i7-1260P
RAM: 32GB
Operating system: Ubuntu 22.04.4

These aren't complex analyses so any computer should work

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
