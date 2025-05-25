# Nucleus Segmentation Based on Layout Prompts and Text Prompts

We designed two optional layout prompts generation modes: manual and automatic. These modes are intended to meet different task requirements.

## YOLO11

To ensure sufficient inference speed in the automatic layout prompt mode, we adopted YOLO11n, a lightweight object detection model from the YOLO11 series.

### Dataset Preparation

#### Prerequisite Steps to Execute

**trans.py**: Convert XML annotations to YOLO-style annotations, retaining only the bounding boxes.
**transtopng.py**: Convert the original images from TIF format to PNG format.
**split.py**: Split the dataset into training and validation sets. And create data.yaml file.

#### Training

**train.py**: Train YOLO11.
**val.py**: Validate YOLO11.

#### Predict

**predict.ipynb**: Predict on MoNuSeg test data. See Precision, Recall, F1-score, mAP50, mAP50-95
**predict.py**: Predict on a single image of MoNuSeg test data.

```bash
Returns:
        list: if isbox=True, return bounding box coordinates list [[x1,y1,x2,y2], ...]
              if isbox=False, return center point coordinates list, each element shape is (1,2)
```

## Citing

- YOLO11 [[Code](https://github.com/ultralytics/ultralytics)]