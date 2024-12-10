import itk
import os
import numpy as np
import random
import cv2
import sys

def read_masks_and_image(binary_mask_path: str, original_image_path: str):
    binary_mask = itk.imread(binary_mask_path, itk.UC)
    original_image = itk.imread(original_image_path, itk.RGBPixel[itk.UC])

    return binary_mask, original_image

def apply_blur(image, kernel_size=3):
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be an odd integer greater than or equal to 1")

    copy = itk.image_duplicator(image)
    size = image.GetLargestPossibleRegion().GetSize()
    height, width = size[1], size[0]
    pad = kernel_size // 2

    for y in range(height):
        for x in range(width):
            r_sum, g_sum, b_sum = 0, 0, 0
            count = 0
            for ky in range(-pad, pad + 1):
                for kx in range(-pad, pad + 1):
                    ny, nx = y + ky, x + kx
                    if 0 <= ny < height and 0 <= nx < width:
                        r, g, b = image.GetPixel([nx, ny])
                        r_sum += r
                        g_sum += g
                        b_sum += b
                        count += 1
            r_avg, g_avg, b_avg = int(r_sum / count), int(g_sum / count), int(b_sum / count)
            copy.SetPixel([x, y], itk.RGBPixel[itk.UC]([r_avg, g_avg, b_avg]))

    # itk.imwrite(copy, 'blurred.png')
    return copy

def apply_color_jitter(image, brightness=.3, contrast=0, saturation=0, hue=[-20, 20]):
    copy = itk.image_duplicator(image)
    size = copy.GetLargestPossibleRegion().GetSize()
    height, width = size[1], size[0]

    brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    hue_adjustment = random.uniform(hue[0], hue[1])

    for y in range(height):
        for x in range(width):
            r, g, b = copy.GetPixel([x, y])
            rgb = np.array([r, g, b])

            # brightness
            rgb = np.clip(rgb * brightness_factor, 0, 255)

            # contrast
            midpoint = 128
            rgb = np.clip((rgb - midpoint) * contrast_factor + midpoint, 0, 255)

            pixel = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
            hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
            h, s, v = hsv_pixel[0, 0]

            # saturation
            s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)

            # hue
            h = (h + hue_adjustment) % 180

            adjusted_hsv_pixel = np.uint8([[[h, s, v]]])
            adjusted_pixel = cv2.cvtColor(adjusted_hsv_pixel, cv2.COLOR_HSV2BGR)
            rgb = np.array(adjusted_pixel[0, 0])

            r, g, b = rgb

            copy.SetPixel([x, y], itk.RGBPixel[itk.UC]([int(r), int(g), int(b)]))
    # itk.imwrite(copy, 'jittered.png')
    return copy

# num_bits is the number of MSB to keep
def apply_random_posterize(image, num_bits=4, probability=0.5):
    # if random.random() > probability:
    #     print('did not make copy')
    #     return image
    
    copy = itk.image_duplicator(image)

    size = image.GetLargestPossibleRegion().GetSize()
    height, width = size[1], size[0]
    
    for y in range(height):
        for x in range(width):
            r, g, b = image.GetPixel([x, y])
            r = (r >> (8 - num_bits)) << (8 - num_bits)
            g = (g >> (8 - num_bits)) << (8 - num_bits)
            b = (b >> (8 - num_bits)) << (8 - num_bits)
            copy.SetPixel([x, y], itk.RGBPixel[itk.UC]([r, g, b]))
    # itk.imwrite(copy, 'posterized.png')
    return copy

def apply_random_adjust_sharpness(image, sharpness_factor=2, probability=0.5):
    # if random.random() > probability:
    #     print("Sharpness adjustment skipped.")
    #     return image

    blurred = apply_blur(image, kernel_size=3)

    size = image.GetLargestPossibleRegion().GetSize()
    height, width = size[1], size[0]
    copy = itk.image_duplicator(image)

    for y in range(height):
        for x in range(width):
            r_orig, g_orig, b_orig = image.GetPixel([x, y])

            r_blur, g_blur, b_blur = blurred.GetPixel([x, y])

            r = np.clip((sharpness_factor * r_orig) - ((sharpness_factor - 1) * r_blur), 0, 255)
            g = np.clip((sharpness_factor * g_orig) - ((sharpness_factor - 1) * g_blur), 0, 255)
            b = np.clip((sharpness_factor * b_orig) - ((sharpness_factor - 1) * b_blur), 0, 255)

            copy.SetPixel([x, y], itk.RGBPixel[itk.UC]([int(r), int(g), int(b)]))

    # itk.imwrite(copy, 'adjusted_sharpness.png')
    return copy

# get the roi given the binary mask
def get_roi(binary_mask, original_image):
    roi = itk.image_duplicator(original_image)
    mask_arr = itk.array_from_image(binary_mask)
    for i in range(len(mask_arr)):
        for j in range(len(mask_arr[0])):
            if mask_arr[i][j] < 127:
                roi.SetPixel((j, i), itk.RGBPixel[itk.UC]([0, 0, 0]))
    # itk.imwrite(roi, 'roi.png')
    return roi

# inverts the binary mask
def inverse_of_mask(binary_mask):
    inverted_arr = itk.array_from_image(binary_mask)
    for i in range(len(inverted_arr)):
        for j in range(len(inverted_arr[0])):
            if inverted_arr[i][j] > 127:
                inverted_arr[i][j] = 0
            else:
                inverted_arr[i][j] = 255
    inverted_image = itk.image_from_array(inverted_arr)
    # itk.imwrite(inverted_image, 'inverted.png') 
    return inverted_image

# gets the non region of interest
def get_nonroi(mask_inverse, original_image):
    nonroi = itk.image_duplicator(original_image)
    inverse_arr = itk.array_from_image(mask_inverse)
    for i in range(len(inverse_arr)):
        for j in range(len(inverse_arr[0])):
            if inverse_arr[i][j] == 0:
                nonroi.SetPixel((j, i), itk.RGBPixel[itk.UC]([0, 0, 0]))
    # itk.imwrite(nonroi, 'nonroi.png')
    return nonroi

# combines the augmented roi with the non roi
def combine(roi_augmented, nonroi):
    combined = itk.image_duplicator(nonroi)
    a, b, _ = itk.array_from_image(combined).shape
    for i in range(a):
        for j in range(b):
            if nonroi.GetPixel((j, i)) == itk.RGBPixel[itk.UC]([0, 0, 0]):
                combined.SetPixel((j, i), roi_augmented.GetPixel((j, i)))
    # itk.imwrite(combined, 'combined.png')
    return combined

def main():

    # comment any out if you don't want to test them
    augmentations = [
        lambda img: apply_blur(img, kernel_size=3),
        lambda img: apply_color_jitter(img, brightness=0.5, contrast=0.5, saturation=0.5, hue=[-20, 20]),
        lambda img: apply_random_posterize(img, num_bits=2, probability=1),
        lambda img: apply_random_adjust_sharpness(img, sharpness_factor=3, probability=0.5)
    ]

    if len(sys.argv) != 3:
        print("Usage: python3 augment.py <binary_mask_path> <original_image_path>")
        sys.exit(1)

    binary_mask_path = sys.argv[1]
    original_image_path = sys.argv[2]

    binary_mask, original_image = read_masks_and_image(binary_mask_path, original_image_path)

    roi = get_roi(binary_mask, original_image)
    nonroi = get_nonroi(inverse_of_mask(binary_mask), original_image)

    for a in augmentations:
        roi = a(roi)

    combined = combine(roi, nonroi)
    itk.imwrite(combined, 'result.png')

if __name__ == "__main__":
    main()
