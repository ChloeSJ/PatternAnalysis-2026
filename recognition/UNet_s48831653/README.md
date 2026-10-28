# Evaluation of 2D U-Net vs 2D Convolutional Aggregate Network performance and resource use for the segmentation of HipMRI scans for targeted radiation therapy of the prostate
Data were provided by HipMRI Study for image segmentation - https://arxiv.org/abs/1505.04597

## Overview
A 2D U-Net and 2D convolutional aggregate network (CAN) have been implemented to aid in the segmentation of the prostate in HipMRI scans for the purpose of targeted radiation therapy.
#### 2D U-Net Explanation and Justification of Structure:
The 2D U-Net consists of an encoder and decoder, each with 4 convolution blocks. The convolution block contains 2 3x3 convolutions, the first to alter the number of channels, and the second to refine feature extraction. BatchNorm is used after each convolution since the data is dense and processed in batches of 16. As each sample is normalised, convolutions are performed without bias to prevent creations of redundant variables. ReLU is performed in place after each normalisation to reduce memory consumption. For convolution blocks with high out_channel>512, dropout=0.2 is set to reduce overfitting during training.

Each encoder block is followed by maxPool, reducing the height and width dimensions by 2. As such, each sample is reduced to height and width dimensions (16, 8). The bottleneck performs a final channel expanding convolution block such that each sample has 1024 channels.

The 4 decoder blocks perform a transpose convolution, expanding the sample's height and width by 2, before concatenating the equivalent sample from the encoder leve. This is followed by the convolution block, halving the number of channels. A final 1x1 convolution is performed so that the output has the same number of channels as there are segmentation categories.

**U-Net Model Structure:**
![alt text](UNet.png)

#### 2D CAN Explanation and Justification of Structure:
The 2D CAN **todo description of algorithm**
**todo diagram**

## Dependencies
This repository was trained and evaluated on the Rangpur HPC cluster with the following dependencies:
- python=3.11.15
- numpy=2.4.6
- nibabel=5.4.2
- tqdm=4.70.1 
- torch=2.7.1+cu118
- torchvision=0.22.1+cu118 
- matplotlib=3.11.1 

The results discussed in this report were produced without setting seed. If desired, the seeds for python, numpy and torch can be set for reproducible results.

## Example inputs, outputs, and visual plots of your algorithm
**todo**

## Description of any specific pre-processing used, with references if any
**todo**
**todo** justify training, validation, testing splits of the data

## Investigation of the Open Research Dilemma
**todo**
quantitative benchmarking of your model against the baseline under identical evaluation splits
- resource profiling table documenting computational efficiency (peak GPU VRAM, parameter count, inference runtime/latency)
- in depth qualitative error autopsies analysing 3-5 representative failure cases from the dataset (diagnosing root triggers + failure modes)
- actionable engineering recommendation + trade off synthesis for your project manager (transformations?)


## Decisions regarding dataloading:
- Patient-level splitting (patient data is present in only one of training/validation/testing) of provided data is preserved by loading pre-defined 2D sets
- Image datasets are normalised upon loading as specified in data pipeline guidelines
- Mask datasets are specified as categorical upon loading, converted to [N, H, W, C] vs image sets which are converted to [N, H, W]
- (256, 144) sized images were cropped to (256, 128) so that all images are of the same size, 128 is a power of 2 suitable for U-Net convolution
- drop_last set to True for the training DataLoader for consistent batch sizes to avoid overweighting the final samples and for stable normalisation

todo: num_workers value, pre-fetch factor, persistant worker, transformations

## Decisions regarding loss calculation:
- Combination of weighted binary cross entropy + dice used

## Decisions regarding training + validation:
- 


## Artificial Intelligence Usage Disclosure:
Gemini used:
- for understanding Nifti loading functions provided in the task
- in data_preprocess() for locally saving ndarrays produced by load_data_2D() and accessing this saved data in get_dataloaders()
- for crop2D() to resize (256, 144) images to (256, 128)
- to confirm convolution input/output channel counts for 2D U-Net modules
- suggested use of F.interpolate in UpConv module in forward() for matching sample shapes



## Feasibility Review: 2D vs 3D U-Net for HipMRI prostate segmentation

These models are intended for use by Radiation Oncologists and Medical Physicists to aid in the segmentation of the body outline, bone, bladder, rectum and regions of the prostate within MRI slices or volumes, for the purpose of targeted radiation therapy.

The models will be determined acceptable according to the following criteria:
1.	Advanced model produces an improved Dice score for the segmentation results in the prostate channel (at minimum, > 0.75 dice for 2D U-Net and >0.7 dice for 3D U-Net in the prostate segmentation channel)
2.	Reasonable model training and validation time (<2min per epoch for 2D model, <5min per epoch for 3D model)
3.	Reasonable prediction time (<0.5s per slice for 2D U-Net, 30s for 3D volume)
4.	Peak VRAM < 12GB

The HipMRI keras_slices_data are used for 2D segmentation and were confirmed to have patient-level splitting between the training, validation and testing groups. 2D data loaders have been created, containing tensors with dimension [B, C, H, W] MRI image and segmentation mask tensor slices. Specifically, this 2D data is formatted [B, 5, 256, 128], with channels corresponding to background/rectum, soft tissue, bone, bladder, and prostate.

The basic model implementation is the easy level 2D U-Net with 4 3x3 convolution encoder blocks with conv, norm, ReLU layers followed a max pool to decrease the HxW dimensions and increase channels. The bottleneck is the same 3x3 convolution block, reaching [B, 1024, 16, 8]. This is followed by 4 decoding blocks, each performing a 2x2 transpose convolution to expand HxW dimensions, concatenates the skip from the corresponding encoding layer and then the 3x3 convolution block. A final 1x1 convolution is used to produce the predicted segmentation mapping. The results of this model are as follows:
-	Training and validation time is approximately 49s per epoch, peak VRAM = 3.1845GB, final dice for the prostate channel = 0.9108
-	Initial testing time is approximately 0.27s per HipMRI Slice with peak VRAM = 1.1256 GB and final dice for the prostate channel = 0.8627. The segmentation predictions have been visualised for a full volume of slices for observation and have produced results similar to the true segments

3D data loaders contain tensors with dimension [B, C, D, H, W], the dimension D and confirmation of the number of channels are yet to be implemented. The improved model will be the normal level 3D U-Net, to determine whether the improved accuracy discussed in lectures, as a result of the context provided by 3D volumes is worth the increase in VRAM and training time that comes with a practical implementation. Singh et al. (2025) and Woo and Lee (2021) have investigated this in regard to the segmentation of brain MRI scans and independently determined 3D U-Net was more accurate with less training samples but was limited by its memory requirements.

Therefore, the main risk for the reminder of the project is memory overflow when training the 3D U-Net. To mitigate this risk, any execution on the Rangpur cluster will be limited to 2 hours and 12GB VRAM, smaller batch sizes will be used and section of the volume may be evaluated instead.

The next steps to be followed are implementing graphing the validation and training losses together for the 2D U-Net, implementing data loading for 3D MRI volumes, building the 3D U-Net module training, validating, and testing it followed by comparisons to the 2D U-Net results. 

If the 3D U-Net segmentation model cannot be implemented, a 2D improved U-Net will instead be used and compared to the standard U-Net with stricter acceptance criteria.

**References:**

- Singh, A., Sharma, A., & Abdul, A. (2025). 3D u-net vs. 2D u-net for brain MRI segmentation: A comparative study on accuracy and efficiency. Journal of Basic Science and Engineering, 22(1), 487. https://doi.org/10.10399/JBSE.2025237518 

- Woo, B., & Lee, M. (2021). Comparison of tissue segmentation performance between 2D u-net and 3D u-net on brain MR images. 2021 International Conference on Electronics, Information, and Communication (ICEIC), 1–4. https://ieeexplore.ieee.org/document/9369797 