# Data_preprocess_pipeline

## Transcriptomics data acquisition and processing

<div align="justify">

Brain-wide transcriptomics data are sourced from the Allen human brain atlas (AHBA, (https://human.brain-map.org/)), including over 58k probes sampled across 3,702 brain locations from six donors.  
We use the abagen toolbox<sup>[1]</sup> for data preprocessing and employ the AAL atlas to map brain locations to ROIs, resulting in a number of 15,633 genes expressed across 116 ROIs.  
To ensure that the transcriptomic knowledge closely aligns with the underlying mechanisms of the corresponding brain diseases, we further filter the genes to focus on those associated with disease risk. Specifically, we utilize two large-scale meta-GWAS results: one from AD (21,982 AD cases, 41,944 controls) from the International Genomics of Alzheimer's Project (IGAP)<sup>[2]</sup> and another for ADHD<sup>[3]</sup> (20,183 ADHD cases, 35,191 controls).  
We use the MAGMA<sup>[4]</sup> to derive gene-level **$p$**-values from SNP-level meta-GWAS results and keep nominally significant ones (i.e., **$p<0.05$**). This process yields a total of 1,216 genes expressed across 116 ROIs for AD and 1,524 genes expressed across the same 116 ROIs for ADHD. These expression datasets are used to construct the corresponding T-RRI edge matrices for the AD and ADHD prediction tasks, respectively.

</div>

## Imaging data acquisition and processing

<div align="justify">

We utilized two brain disorder datasets in our study two brain disease cohorts (ADNI and ADHD-200) involving four imaging modalities (AV45-PET, FDG-PET, VBM-sMRI, and fMRI) to evaluate the performance of STMM. The acquisition and preprocessing of each modality are described as follows.

</div>

### AV45-PET modality

<div align="justify">

The preprocessed [18F]Florbetapir PET (AV45) scans were obtained from the LONI database (https://adni.loni.usc.edu/). Prior to download, all PET images underwent standard preprocessing procedures as described by Jagust et al.<sup>[5]</sup> and Yan et al.<sup>[6]</sup>, including averaging, alignment to a standard anatomical space, resampling to a common voxel grid, smoothing to achieve a uniform resolution, and normalization to a cerebellar gray matter reference region, resulting in standardized uptake value ratio (SUVR) images. After download, each PET image was rigidly aligned to the corresponding T1-weighted MRI scan acquired at the same visit. The images were then spatially normalized to the Montreal Neurological Institute (MNI) space using transformation parameters derived from MRI segmentation, with a final voxel resolution of 2 × 2 × 2 mm. Regional amyloid burden values were subsequently extracted at the region-of-interest (ROI) level using the Automated Anatomical Labeling (AAL) atlas implemented in MarsBaR<sup>[7]</sup>.

</div>

### FDG-PET modality

<div align="justify">

The preprocessed [18F]FDG-PET scans were obtained from the LONI database (https://adni.loni.usc.edu/). After download, each scan was aligned to the corresponding T1-weighted MRI acquired at the same visit for each participant. The aligned PET images were then spatially normalized to the MNI space using transformation parameters from the MRI segmentation, with a final voxel resolution of 2 × 2 × 2 mm. Measurements of glucose metabolism were subsequently extracted at the ROI level using the AAL atlas implemented in MarsBaR<sup>[7]</sup>. More details can be found at Yao et al.<sup>[8]</sup>.

</div>

### VBM-sMRI modality

<div align="justify">

To investigate brain-associated MRI imaging phenotypes, we analyzed data from the ADNI database (https://adni.loni.usc.edu/), including MRI scans from participants<sup>[9]</sup>. Each participant underwent at least two baseline MP-RAGE scans at 1.5 T, following the ADNI MRI protocol<sup>[10]</sup>. The scans were processed using voxel-based morphometry (VBM), a widely employed automated MRI analysis method<sup>[11]</sup>.  
VBM analysis was conducted with SPM12, which generated unmodulated, normalized grey matter (GM) density maps with a voxel resolution of 1 × 1 × 1 mm, smoothed using a 10 mm full-width at half-maximum (FWHM) Gaussian kernel. For each participant, two independently processed maps were averaged to obtain a mean GM density map.  
Regional GM density values were extracted in the MNI space using the MarsBaR ROI toolbox<sup>[7]</sup>.  
Both ADNI dataset and ADHD-200 dataset were preprocessed with the same way.

</div>

### fMRI modality

<div align="justify">

The preprocessed fMRI scans were obtained from the NITRC (https://www.nitrc.org/).  
The original fMRI data were preprocessed using a public toolbox named DPABI<sup>[12]</sup> (for Data Processing \& Analysis of Brain Imaging, (http://rfmri.org/dpabi). The preprocessing steps were as follows: (1) remove the first 10 volumes to ensure that the BOLD signal was stable; (2) slice timing, correct the difference due to acquisition times between slices in the volume; (3) head motion correction; (4) normalization, register the data to the EPI standard template and resample it to **$3.0 \times 3.0 \times 3.0$** mm; and (5) spatial smoothing with a 6-mm full width at half maximum (FWHM) Gaussian kernel. Subjects whose head movement exceeded 2.0 mm were excluded.

</div>

<div align="justify">

For the fMRI modality, which includes a temporal dimension, the dynamic imaging data were represented as a three-dimensional tensor **$\mathbf{D}_{dynamic}^{m} = \{\mathbf{X}_1^m, \cdots, \mathbf{X}_n^m\} \in \mathbb{R}^{n \times p \times d}$**, where each sample-specific matrix **$\mathbf{X}_s^m \in \mathbb{R}^{p \times d}$** corresponds to **$p$** time points across **$d$** regions of interest (ROIs). We constructed sample-specific region-wise region-to-region interaction (R-RRI) networks for each individual by computing the Pearson correlation coefficient (PCC) between all pairs of ROIs over time. The resulting correlation matrices were thresholded using a modality-specific hyperparameter **$\lambda_r^m$** to generate a set of binary edge matrices **$\mathcal{E}^m = \{\mathbf{E}_s^m\}_{s=1}^{n}$**. SIP et al.<sup>[13]</sup> demonstrated that underlying dynamics in resting-state fMRI can be effectively characterized as noisy fluctuations around a single fixed point, supporting the utility of simplified temporal representations. In addition, Lee et al.<sup>[14]</sup> evaluated the discriminative power of features generated by principal component analysis (PCA) in distinguishing individuals with psychophysiological insomnia (PI) from healthy controls (HC), and found that PCA-based features achieved the best classification performance across all evaluated metrics. Motivated by these findings, we applied PCA to the temporal signals of each ROI and retained the first principal component to obtain a single representative value, which was used as the node feature in the R-RRI graph.

</div>

---

### References

[1] R. Markello et al., “Standardizing workflows in imaging transcriptomics with the abagen toolbox,” *eLife*, vol. 10, p. e72129, 2021.  
[2] B. W. Kunkle et al., “Genetic meta-analysis of diagnosed Alzheimer’s disease identifies new risk loci and implicates Aβ, tau, immunity and lipid processing,” *Nature Genetics*, vol. 51, no. 3, pp. 414–430, 2019.  
[3] D. Demontis et al., “Discovery of the first genome-wide significant risk loci for attention deficit/hyperactivity disorder,” *Nature Genetics*, vol. 51, no. 1, pp. 63–75, 2019.  
[4] C. A. de Leeuw et al., “MAGMA: generalized gene-set analysis of GWAS data,” *PLoS Computational Biology*, vol. 11, no. 4, p. e1004219, 2015.  
[5] W. J. Jagust et al., “The Alzheimer’s Disease Neuroimaging Initiative positron emission tomography core,” *Alzheimer’s & Dementia*, vol. 6, no. 3, pp. 221–229, 2010.  
[6] J. Yan et al., “Transcriptome-guided amyloid imaging genetic analysis via a novel structured sparse learning algorithm,” *Bioinformatics*, vol. 30, no. 17, pp. i564–i571, 2014.  
[7] N. Tzourio-Mazoyer et al., “Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain,” *NeuroImage*, vol. 15, no. 1, pp. 273–289, 2002.  
[8] X. Yao et al., “Tissue-specific network-based genome wide study of amygdala imaging phenotypes to identify functional interaction modules,” *Bioinformatics*, vol. 33, no. 20, pp. 3250–3257, 2017.  
[9] L. Shen et al., “Whole genome association study of brain-wide imaging phenotypes for identifying quantitative trait loci in MCI and AD: A study of the ADNI cohort,” *NeuroImage*, vol. 53, no. 3, pp. 1051–1063, 2010.  
[10] C. R. Jack Jr et al., “The Alzheimer’s Disease Neuroimaging Initiative (ADNI): MRI methods,” *Journal of Magnetic Resonance Imaging*, vol. 27, no. 4, pp. 685–691, 2008.  
[11] S. L. Risacher et al., “Baseline MRI predictors of conversion from MCI to probable AD in the ADNI cohort,” *Current Alzheimer Research*, vol. 6, no. 4, pp. 347–361, 2009.  
[12] C.-G. Yan et al., “DPABI: Data Processing & Analysis for (Resting-State) Brain Imaging,” *Neuroinformatics*, vol. 14, pp. 339–351, 2016.  
[13] V. Sip et al., “Characterization of regional differences in resting-state fMRI with a data-driven network model of brain dynamics,” *Science Advances*, vol. 9, no. 11, p. eabq7547, 2023.  
[14] M. H. Lee et al., “Multitask fMRI and machine learning approach improve prediction of differential brain activity pattern in patients with insomnia disorder,” *Scientific Reports*, vol. 11, no. 1, p. 9402, 2021.

