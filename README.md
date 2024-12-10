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
This section includes two key scripts:  

1. `Sam2_image_predictor.py`: generates binary segmentation masks for regions of interest using the point-prompting technique (input points and labels). It combines functionality from the SAM2 notebooks: `image_predictor_example.ipynb` and `automatic_mask_generator_example.ipynb`.  

##### Requirements:  
- A CUDA-enabled GPU.  
- SAM2 installed (follow the instructions in the [SAM2 GitHub repository](https://github.com/facebookresearch/sam2)).  
 

##### Usage:  
To run the script, you will need:  
- The SAM2 model checkpoint file: `sam2_hiera_large.pt`.  
- The configuration file: `sam2_hiera_l.yaml`.  

For detailed installation and usage instructions, please refer to the official SAM2 documentation.

Run the script from the terminal using the following command:  
```bash
python sam2_image_predictor.py
```
2. Analyze_mask.py`: Evaluates segmentation results by comparing ground truth masks and SAM2 predictions. It calculates performance metrics, such as the Dice Similarity Coefficient (DSC) and Intersection over Union (IoU).  

##### Features:  
- Does not require a GPU.  
- Can be run locally on your computer.  

##### Usage:  
Run the script from the terminal using the following command:  
```bash
python analyze_mask.py
```

## Watershed Segmentation

The watershed segmentation folder contains three (3) python files:

1. `segment_previous_model.py`: Uses a previously defined segmentation model to perform segmentation. This model struggles with kidney stone dust fragments.
2. `segment_original.py`: This is the complete watershed segmentation including Hough Circle Transform, Sobel Edge Detection, etc. and the resulting image is the original image with a border showing the original image into its respective regions.
3. `segment_watershed_regions`: This is also the complete watershed segmentation including Hough Circle Transform, etc. but the resulting image is the colored regions instead of the original image.

Any of the files can be executed in the same fashion:

`<file.py> <original_image> <binary_mask> <output_image>`

I also included a bash script to show how I iterated through a series of folders and placed the output images into a different folder. This assumes the same file names across all folders. It can be executed using the following:

`./segment.sh <original_image_folder> <binary_mask_folder> <output_folder>`
