import os
from PIL import Image
import numpy as np
import  matplotlib.pyplot as plt
from natsort import natsorted
import json


def dice(gt_mask, pred_mask):
    # Validate that both masks have the same shape
    if gt_mask.shape != pred_mask.shape:
        raise ValueError("Ground truth and predicted mask shapes must be the same")
    
    # Ensure both masks are binary
    gt_mask = (gt_mask > 0).astype(np.uint8)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    intersection = np.logical_and(gt_mask, pred_mask)
    mask_area = np.sum(gt_mask) + np.sum(pred_mask)

    if mask_area == 0:
        return 1.0  
    
    # Calculate the Dice score
    dice = 2 * np.sum(intersection) / mask_area

    return dice



def iou (gt_mask, pred_mask):

    # Validate inputs
    if not isinstance(gt_mask, np.ndarray) or not isinstance(pred_mask, np.ndarray):
        raise TypeError("Both gt_mask and pred_mask must be NumPy arrays.")
    
    # # Validate that both masks have the same shape
    # if gt_mask.shape != pred_mask.shape:
    #     raise ValueError(f"Ground truth shape {gt_mask.shape} and predicted shape {pred_mask.shape} do not match")
    
    # Ensure both masks are binary
    gt_mask = (gt_mask > 0).astype(np.uint8)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)

    if np.sum(union) == 0:
        return 1.0

    iou = np.sum(intersection) / np.sum(union)

    return iou



def analyse_mask (gt_mask, pred_mask):
    
    dice_score = dice(gt_mask, pred_mask)
    iou_score = iou(gt_mask, pred_mask)

    return dice_score, iou_score



def box_plot (dice, iou):
    # Create a box plot for Dice and IoU
    plt.figure(figsize=(10, 6))
    
    # Prepare data for box plot
    data = [dice, iou]
    labels = ["Dice Score", "IoU Score"]
    
    # Create the box plot
    plt.boxplot(data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="skyblue", color="blue"), 
                medianprops=dict(color="red"), 
                whiskerprops=dict(color="blue"), 
                capprops=dict(color="blue"))
    
    # Add labels, title, and grid
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Scores", fontsize=12)
    plt.title("Distribution of Dice and IoU Scores", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Save and show the plot
    plt.savefig("boxplot.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()



def plot_metrics (dice, iou):
    # Convert lists to numpy arrays for easy plotting
    indices = np.arange(len(dice))  # X-axis indices

    # Create a bar plot for Dice and IoU
    bar_width = 0.35  # Width of the bars
    plt.figure(figsize=(10, 6))

    # Plot bars for Dice and IoU
    plt.bar(indices, dice, bar_width, label="Dice Score", color="blue", alpha=0.7)
    plt.bar(indices + bar_width, iou, bar_width, label="IoU Score", color="orange", alpha=0.7)

    # Add labels, title, and legend
    plt.xlabel("Mask Index", fontsize=12)
    plt.ylabel("Scores", fontsize=12)
    plt.title("Dice and IoU Scores for Mask Comparison", fontsize=14)
    plt.xticks(indices + bar_width / 2, [f"{i+1}" for i in indices], rotation=45)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.savefig("graph.png", dpi=300, bbox_inches='tight')
    # Show plot
    plt.tight_layout()
    plt.show()


def main ():

    gt_path = "/home/oguinekj/coursework/Binary"
    pred_path = "/home/oguinekj/coursework/predicted_mask"

    gts = natsorted(os.listdir(gt_path))
    preds = natsorted(os.listdir(pred_path))

    dice_list, iou_list = [], []

    if len(gts) != len(preds):
        raise ValueError (f"Length of gt_masks {len(gts)} must be the same with pred_mask {len(preds)}")

    for gt_, pred_ in zip(gts, preds):
        gt_mask_path = os.path.join(gt_path, gt_)
        pred_mask_path = os.path.join(pred_path, pred_)

        gt_mask = np.array(Image.open(gt_mask_path).convert("L"))
        pred_mask = np.array(Image.open(pred_mask_path).convert("L"))

        # Check shape and resize if needed
        if gt_mask.shape != pred_mask.shape:
            print(f"Shape mismatch: {gt_mask.shape} vs {pred_mask.shape}. Skipping...")
            continue  # Skip this pair or resize

        dice_score, iou_score = analyse_mask(gt_mask, pred_mask)

        dice_list.append(dice_score)
        iou_list.append(iou_score)

    
    # plot metrice
    plot_metrics(dice_list, iou_list)
    box_plot(dice_list, iou_list )


    # find the mean and standard deviation
    mean_dice, dice_std = np.mean(dice_list), np.std(dice_list)
    mean_iou, iou_std = np.mean(iou_list), np.std(iou_list)

    # Save results in a dictionary
    metrics_summary = {
        "Dice": {"mean": mean_dice, "std": dice_std},
        "IoU": {"mean": mean_iou, "std": iou_std},
        "Dices": dice_list,
        "IOUs": iou_list,
    }


    # Dump the dictionary to a JSON file
    json_file_path = "metrics_summary.json"
    with open(json_file_path, "w") as json_file:
        json.dump(metrics_summary, json_file, indent=4)

    print(f"Metrics summary saved to {json_file_path}")


if __name__ == "__main__":
    main()