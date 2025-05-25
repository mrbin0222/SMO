import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_dice(pred, target):
    """
    calculate DICE score
    Args:
    pred: predicted mask
    target: target mask
    """
    smooth = 1e-5
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def calculate_miou(pred, target, num_classes):
    """
    calculate mIoU
    Args:
    pred: predicted mask
    target: target mask
    num_classes: number of classes
    """
    # flatten predicted and target masks
    pred = pred.flatten()
    target = target.flatten()
    
    # calculate confusion matrix
    cm = confusion_matrix(target, pred, labels=range(num_classes))
    
    # calculate IoU for each class
    iou_list = []
    for i in range(num_classes):
        if cm[i, i] == 0:
            iou_list.append(0)
            continue
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        iou = intersection / (union + 1e-10)
        iou_list.append(iou)
    
    # calculate mIoU
    miou = np.mean(iou_list)
    return miou

# # example usage
# if __name__ == "__main__":
#     # assume we have a binary classification problem
#     pred_mask = np.array([[1, 0], [0, 1]])
#     target_mask = np.array([[1, 0], [0, 1]])
    
#     # calculate DICE
#     dice_score = calculate_dice(pred_mask, target_mask)
#     print(f"DICE Score: {dice_score:.4f}")
    
#     # calculate mIoU
#     miou_score = calculate_miou(pred_mask, target_mask, num_classes=2)
#     print(f"mIoU Score: {miou_score:.4f}")
