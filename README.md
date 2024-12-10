# CS-6357-01-ROI-Based-Augmentations-and-SAM2

The code contains three folders, each correlating to one portion of the project. These run independently of one another. 

1. ROI-Based Augmentations
2. SAM2 Segmentation
3. Watershed Segmentation

> Let me preface by saying **I cannot share the data** so you will not be able to execute this program on our dataset. That said, to run these, you'll go into the respective folder and execute the specific command.

## ROI-Based Augmentations

ROI-based augmentations contains a single `augment.py` file used for augmenting a single image. It takes in a single binary image and an original image then executes pixel-by-pixel augmentations and outputs a combined image.

`augment.py <binary_mark> <original_image>`

It outputs to a file named `result.png` in the current folder.

## SAM2 Segmentation



## Watershed Segmentation

The watershed segmentation folder contains three (3) python files:

1. `segment_previous_model.py`: Uses a previously defined segmentation model to perform segmentation. This model struggles with kidney stone dust fragments.
2. `segment_original.py`: This is the complete watershed segmentation including Hough Circle Transform, Sobel Edge Detection, etc. and the resulting image is the original image with a border showing the original image into its respective regions.
3. `segment_watershed_regions`: This is also the complete watershed segmentation including Hough Circle Transform, etc. but the resulting image is the colored regions instead of the original image.

Any of the files can be executed in the same fashion:

`<file.py> <original_image> <binary_mask> <output_image>`

I also included a bash script to show how I iterated through a series of folders and placed the output images into a different folder. This assumes the same file names across all folders. It can be executed using the following:

`./segment.sh <original_image_folder> <binary_mask_folder> <output_folder>`
