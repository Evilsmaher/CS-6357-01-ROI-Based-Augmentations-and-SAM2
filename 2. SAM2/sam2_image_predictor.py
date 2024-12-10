# Some part of this script is adapted from the automatic_mask_generator_example.ipynb and image_predictor_example.ipynb scripts available in the Meta Segment Anything Model repository https://github.com/facebookresearch/sam2/tree/main/notebooks. For more details, visit the repository:https://github.com/facebookresearch/sam2.

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from natsort import natsorted

# import sam2 independencies
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import torch
print(torch.cuda.is_available())

# Device setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda")
   
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

# Helper functions
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, filename, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f} for image:{filename}", fontsize=18)  #edited
        plt.axis('off')
        plt.show()


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def erode_masks(image):
    kernel = np.ones((5,5), np.uint8)
    erroded_image =cv2.erode(image, kernel, iterations=1)
    return erroded_image


def predict_mask (predictor, image, filename, save_path): 

    # initialize predictor
    predictor.set_image(image)

    # add initial single positive point input 
    input_point = np.array([[453, 410]])
    input_label = np.array([1])

    # Predict masks
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    # visualize mask from  intial input points
    show_masks(image, masks, scores, filename, point_coords=input_point, input_labels=input_label, borders=True)

    # use get the mask with best logits and use it to refine the multi point inputs
    use_mask = logits[np.argsort(scores)[-2], :, :]
    mask_input=use_mask[None, :, :]

    # Adding background point. modify points and labels according to image.
    # '1' for foreground label, '0' for background label
    # no of positive and negative points must corresponds to the labels
    input_points = np.array([[34, 187], [211, 362], [403,105], [232, 98],   [53, 85],[394, 241], [456, 301]])
    input_labels = np.array([1, 1, 1, 1,   0, 0, 0])  

    # Predict masks, scores, and logits
    masks2, scores2, logits2 = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        mask_input=mask_input,  
        multimask_output=True,  
    )

    show_masks(image, masks2, scores2, filename, point_coords=input_point, input_labels=input_label, borders=True)

    # Obtain the mask with the logit that provides the best segmentation, which may not always correspond to the highest logit value.
    sorted_indices2 = np.argsort(scores2)[::-1]  # Indices in descending order of scores - highest to lowest
    use_index = sorted_indices2[0]  #  Adjust the index based on the visualization of the best mask segmentation.
    use_mask2 = masks2[use_index]
    binary_masks = (use_mask2 > 0).astype(np.uint8) * 255

    # Optional: save mask if needed for further experiments
    cv2.imwrite((os.path.join(save_path, filename)), binary_masks)

    return binary_masks



def mask_generator(sam2, image):
    
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=128, 
        points_per_batch=128,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.5,
        stability_score_offset=0.7,
        crop_n_layers=2,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=4,
        min_mask_region_area=1.0,
        use_m2m=True,
    )
   
    masks2 = mask_generator.generate(image)

    # Optional: show SAM2 generated masks
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks2)
    plt.axis('off')
    plt.show() 

    return masks2


def extract_masks (masks2, binary_masks, image_name, save_path):
    for i in range (len(masks2)):
        masks2[i]["segmentation"].astype(np.uint8)  
        # Convert mask segmentation to binary
        bin_mask = masks2[i]["segmentation"].astype(np.uint8)
        bin_mask = (bin_mask > 0).astype(np.uint8)  
        segmentation_mask = bin_mask * 255 

        # Access the area of the mask
        mask_area = masks2[i]["area"]
            
        if mask_area <= 100:
            erode_masks(segmentation_mask)  # erode mask to remove mask boundary uncertainty
            binary_masks += segmentation_mask

    file_path = os.path.join(save_path, image_name)
    cv2.imwrite(file_path, binary_masks)
        
            
    
def main():
    save_path = "path_to_directory_to_save_segementation mask"
    image_path = "path_to_the_images"

    # initialize SAM2 model with config file path and checkpoint path
    sam2_checkpoint = "....../sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    # load and sort images from directory
    images = natsorted(os.listdir(image_path))

    for image_file in images: 

        # Get image path
        image_path = os.path.join(image_path, image_file)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        # load predictor
        predictor = SAM2ImagePredictor(sam2)

        initial_mask = predict_mask(predictor, image, image_file)
        masks2 = mask_generator(sam2, image)
        extract_masks (masks2, initial_mask, image_file, save_path) 
        

        
if __name__=="__main__":
    main()

