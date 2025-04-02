# STMM: Sample-aware Trustworthy Multi-View Multi-Modal Fusion for Diagnosis of Brain Disorders with Imaging Transcriptomics

# Overview
Brain transcriptomics provides insights into the molecular mechanisms by which the brain coordinates its functions and processes. However, existing multimodal methods for brain disease prediction mainly rely on imaging and sometimes genetic data, often neglecting the transcriptomic basis. Moreover, most studies overlook both modality informativeness and its intra-sample disparities. We propose STMM, a sample-aware trusted multiview multimodal framework that integrates limited yet informative brain-wide transcriptomics with imaging data. STMM constructs view-specific brain regional co-function networks, employs graph attention and cross-modal attention for representation and fusion, and introduces a novel true-false-harmonized class probability strategy for adaptive confidence refinement. Experiments on ADNI and ADHD-200 datasets demonstrate STMM’s superior performance and its ability to uncover meaningful brain biomarkers.

![STMM_framework](STMM_framework.png)

## Requirements

- Python 3.6
- PyTorch 1.10.2
- PyTorch Geometric
- scikit-learn
- numpy

## Data Preparation
The data used can be obtained from ADNI and ADHD-200. We provide the data of NC vs. AD.

## Disclaimer
This tool is for research purposes and not approved for clinical use.

## Acknowledgments
This tool is developed in Yao Lab. We thank all the contributors and collaborators for their support.

# Ciatation



