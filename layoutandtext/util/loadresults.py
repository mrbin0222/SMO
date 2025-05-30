import json
import os

import numpy as np


def load_masks_and_scores(image_name, load_dir='results'):
    """
    Load masks and scores from local directory
    
    Args:
        image_name: name of the image (without extension)
        load_dir: directory where results are saved
    
    Returns:
        n_masks: list of dictionaries containing mask information
        n_scores: list of scores
    """
    # Check if files exist
    masks_file = os.path.join(load_dir, f'{image_name}_masks.json')
    scores_file = os.path.join(load_dir, f'{image_name}_scores.json')
    
    if not os.path.exists(masks_file) or not os.path.exists(scores_file):
        raise FileNotFoundError(f"Results for {image_name} not found in {load_dir}")
    
    # Load masks
    with open(masks_file, 'r') as f:
        loaded_masks = json.load(f)
    
    # Convert loaded masks back to numpy arrays
    n_masks = []
    for mask in loaded_masks:
        converted_mask = {
            'segmentation': np.array(mask['segmentation'], dtype=bool),
            'area': np.int64(mask['area']),
            'bbox': np.array(mask['bbox'], dtype=np.int64),
            'predicted_iou': np.float64(mask['predicted_iou']),
            'point_coords': np.array(mask['point_coords'], dtype=np.float64),
            'stability_score': np.float64(mask['stability_score']),
            'crop_box': np.array(mask['crop_box'], dtype=np.int64)
        }
        n_masks.append(converted_mask)
    
    # Load scores
    with open(scores_file, 'r') as f:
        n_scores = np.array(json.load(f), dtype=np.float64)
    
    print(f"Successfully loaded results for {image_name}")
    print(f"Number of masks: {len(n_masks)}")
    print(f"Number of scores: {len(n_scores)}")
    
    return n_masks, n_scores

# Example usage:
# n_masks, n_scores = load_masks_and_scores("image001")
