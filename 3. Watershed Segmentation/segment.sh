#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <original_image_folder> <binary_image_folder> <output_folder>"
    exit 1
fi

# Assign arguments to variables
original_folder=$1
binary_folder=$2
output_folder=$3

# Create the output folder if it does not exist
mkdir -p "$output_folder"

# Iterate through the images in the original folder
for original_image in "$original_folder"/*; do
    # Get the filename of the current image (without path)
    image_name=$(basename "$original_image")

    # Construct the corresponding binary image path
    binary_image="$binary_folder/$image_name"

    # Check if the binary image exists
    if [ -f "$binary_image" ]; then
        # Construct the output image path
        output_image="$output_folder/$image_name"

        # Call the Python script with the appropriate arguments
        python segment_truth.py "$original_image" "$binary_image" "$output_image"

        echo "Processed $original_image -> $output_image"
    else
        echo "Warning: No corresponding binary image found for $original_image"
    fi
done

echo "Processing complete."
