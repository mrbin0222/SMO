import numpy as np
import torch


def calculate_metrics(pred, target):
    """
    calculate DICE and mIoU metrics for segmentation tasks
    
    Args:
        pred: predicted result, shape of (N, H, W), can be numpy array or torch tensor
        target: target mask, shape of (N, H, W), can be numpy array or torch tensor
    
    Returns:
        dice: DICE coefficient
        miou: mIoU value
    """
    # check input type and convert to numpy array
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # binarize predicted result
    pred = (pred > 0.5).astype(np.float32)
    
    # calculate intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    # calculate DICE coefficient
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
    
    # calculate IoU
    iou = intersection / (union + 1e-6)
    
    return float(dice), float(iou)


