Data were provided by HipMRI Study for image segmentation - https://arxiv.org/abs/1505.04597

## Decisions regarding dataloading:
- Patient-level splitting (patient data is present in only one of training/validation/testing) of provided data is preserved by loading pre-defined 2D sets
- Image datasets are normalised upon loading as specified in data pipeline guidelines
- Mask datasets are specified as categorical upon loading, converted to [N, H, W, C] vs image sets which are converted to [N, H, W]
- (256, 144) sized images were cropped to (256, 128) so that all images are of the same size, 128 is a power of 2 suitable for U-Net convolution
- drop_last set to True for the training DataLoader for consistent batch sizes to avoid overweighting the final samples and for stable normalisation

todo: num_workers value, pre-fetch factor, persistant worker, transformations

## Decisions regarding the 2D U-Net modules:
- Encoder and decoder consist of 4 convolution blocks, each with 2 3x3 convolutions. The first expands the number of channels, the second is to refined feature extraction
- Convolution blocks use a batchNorm after each convolution since the 2D U-Net is using batchSize=16. As normalisation is performed after the convolution, each convolution has bias set to false to prevent redundant variables.
- In place ReLU used to reduce memory consumption
- Encoder follows each convolution block with a maxpooling operation, so that the height and width dimensions of each sample are reduced by 2, such that the bottleneck is provided samples with dimensions 16x8
- Bottleneck concists of a convolution block so that the decoder is given 1024 channels per sample
- The 4 decoder blocks perform a transpose convolution to expand the sample dimensions by 2 before concatenating the equivalent sample from the encoder levels. This is followed by a concatenation block as described above where the number of channels is decreased by 2.
- A final 1x1 convolution is performed on all samples so that the output number of channels is equal to the number of segmentation categories

## Artificial Intelligence Usage Disclosure:
Gemini used:
- for understanding Nifti loading functions provided in the task
- in data_preprocess() for locally saving ndarrays produced by load_data_2D() and accessing this saved data in get_dataloaders()
- for crop2D() to resize (256, 144) images to (256, 128)
- to confirm convolution input/output channel counts for 2D U-Net modules
- suggested use of F.interpolate in UpConv module in forward() for matching sample shapes
