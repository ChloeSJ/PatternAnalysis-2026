Data were provided by HipMRI Study for image segmentation - https://arxiv.org/abs/1505.04597

## Decisions regarding dataloading:
- Patient-level splitting (patient data is present in only one of training/validation/testing) of provided data is preserved by loading pre-defined 2D sets
- Image datasets are normalised upon loading as specified in data pipeline guidelines
- Mask datasets are specified as categorical upon loading, converted to [N, H, W, C] vs image sets which are converted to [N, H, W]
- dataset transformations? (same transform applied so that spatial alignment between images and masks are preserved)
- (256, 144) sized images were cropped to (256, 128) so that all images are of the same size, 128 is a power of 2 suitable for U-Net convolution
- drop_last set to True for the training DataLoader for consistent batch sizes to avoid overweighting the final samples and for stable normalisation
- todo: num_workers value, pre-fetch factor, persistant worker, transformations

## Artificial Intelligence Usage Disclosure:
Gemini used:
- for understanding Nifti loading functions provided in the task
- in data_preprocess() for locally saving ndarrays produced by load_data_2D() and accessing this saved data in get_dataloaders()
- for crop2D() to resize (256, 144) images to (256, 128)
