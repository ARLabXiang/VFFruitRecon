# VF-FruitRecon: 3D Fruit Reconstruction and Size Estimation with Foundation Models from Stereo Pair Images.

## Overview
`VF_FruitRecon_directory.py` parses directory of left and right images through Grounded-SAM -> FoundationStereo -> SAM-3D to generate the mesh of individual fruits.

`mask_quality.py` parses the Grounded-SAM outputs against the annotated masks to compute the iou and the matched masks. The `*_masked.png` images are to ignore the blacked out regions as they were not annotated on purpose.

`SizeEstimate.py` produces the size estimation of the mesh and from the baseline models. It will also use the output of `mask_quality.py` to align with the measured sizes.

`plot_result.py` will generate the analysis in the paper from the output of `SizeEstimate.py`.

## Installation
Follow installation in the three submodules (SAM-3D, Grounded-SAM, and Foundation Stereo).

## Datasets
Greenhouse dataset (50 images): [Box](https://cornell.box.com/s/ky38g6ic3tu1jg2ua3k5xb9yr673uzol)

The stereo images are under `sample_image/r` and `sample_image/l`. The `sample_image/gt_mask` folder is primarily for evaluation to match the predicted mask with annotated mask.

Synthetic dataset (Soon)