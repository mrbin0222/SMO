'''n_masks
[{'segmentation': array([[False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         ...,
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False]]),
  'area': 1895,
  'bbox': [1092, 872, 79, 31],
  'predicted_iou': 0.9576271772384644,
  'point_coords': [[1120.0, 870.0],
   [1090.0, 898.0],
   [1150.0, 880.0],
   [1093.0, 897.0],
   [1091.0, 894.0],
   [1109.0, 875.0],
   [1171.0, 901.0],
   [1144.0, 877.0],
   [1112.0, 876.0]],
  'stability_score': 0.9576271772384644,
  'crop_box': [0, 0, 1224, 904]},
 {'segmentation': array([[False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         ...,
...
   [1040.0, 686.0],
   [1041.0, 704.0],
   [1041.0, 687.0]],
  'stability_score': 0.8624655604362488,
  'crop_box': [0, 0, 1224, 904]}]'''

'''n_scores
[tensor(0.7020),
 tensor(0.5405),
 tensor(0.6049),
 tensor(0.5066),
 tensor(0.4972),
 tensor(0.5896),
 tensor(0.6095),
 tensor(0.6293)]'''

import json
import os
from datetime import datetime

import numpy as np
import torch


def save_masks_and_scores(n_masks, n_scores, image_name, save_dir='results'):
    """
    Save masks and scores to local directory using image name
    
    Args:
        n_masks: list of dictionaries containing mask information
        n_scores: list of tensor scores
        image_name: name of the image (without extension)
        save_dir: directory to save results
    """
    # Create save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Convert masks to serializable format
    serializable_masks = []
    for mask in n_masks:
        serializable_mask = {
            'segmentation': mask['segmentation'].tolist(),
            'area': int(mask['area']),  # Convert numpy.int64 to Python int
            'bbox': [int(x) for x in mask['bbox']],  # Convert numpy.int64 to Python int
            'predicted_iou': float(mask['predicted_iou']),
            'point_coords': [[float(x) for x in point] for point in mask['point_coords']],  # Convert numpy.float64 to Python float
            'stability_score': float(mask['stability_score']),
            'crop_box': [int(x) for x in mask['crop_box']]  # Convert numpy.int64 to Python int
        }
        serializable_masks.append(serializable_mask)
    
    # Convert scores to list of floats
    serializable_scores = [float(score.item()) for score in n_scores]
    
    # Save masks
    masks_file = os.path.join(save_dir, f'{image_name}_masks.json')
    with open(masks_file, 'w') as f:
        json.dump(serializable_masks, f, indent=2)
    
    # Save scores
    scores_file = os.path.join(save_dir, f'{image_name}_scores.json')
    with open(scores_file, 'w') as f:
        json.dump(serializable_scores, f, indent=2)
    
    print(f"Results saved to {save_dir}")
    print(f"Masks saved to: {masks_file}")
    print(f"Scores saved to: {scores_file}")

# Example usage:
# save_masks_and_scores(n_masks, n_scores, "image001")

