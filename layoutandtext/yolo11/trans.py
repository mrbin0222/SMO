import json
import os
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
from PIL import Image, ImageDraw

# load monuseg training set
# image path, annotation path
ann_path="MoNuSegTrainingData/Annotations/"
# image path, tif file
im_path="MoNuSegTrainingData/TissueImages/"
# mask path, png file
# label_path="MoNuSeg2018TrainingData/labelcol/"

# list all images
img_res=os.listdir(im_path)
img_res=[s for s in img_res if "tif" in s]
img_res=sorted(img_res)

# list all xml annotations
ann_xml=os.listdir(ann_path)
ann_xml=[s for s in ann_xml if "xml" in s]
ann_xml=sorted(ann_xml)
#ann_xml


def convert_to_yolo_format(box, img_width=1000, img_height=1000):
    """
    convert bounding box to YOLO format
    box: [x1, y1, x2, y2] format bounding box
    return: [x_center, y_center, width, height] normalized YOLO format
    """
    x1, y1, x2, y2 = box
    
    # calculate center point coordinates
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    
    # calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # normalize
    x_center = x_center / img_width
    y_center = y_center / img_height
    width = width / img_width
    height = height / img_height
    
    return [x_center, y_center, width, height]

# create directory to save YOLO format annotations
yolo_labels_dir = "MoNuSegTrainingData/yololabel/"
os.makedirs(yolo_labels_dir, exist_ok=True)

# process all images
for index in range(len(img_res)):
    # parse XML file
    tree = ET.parse(ann_path + ann_xml[index])
    root = tree.getroot()
    
    # get all segmentation points
    ptss = []
    for p in root.findall(".//Region"):
        pt = []
        for q in p.findall(".//Vertex"):
            x = float(q.get("X"))
            y = float(q.get("Y"))
            a = int(x)
            b = int(y)
            if int(x) >= 1000:
                a = 999
            if int(y) >= 1000:
                b = 999
            pt.append([a, b])
        pt = np.array(pt).astype(np.int32)
        pt = pt.reshape((pt.shape[0], 1, pt.shape[1]))
        ptss.append(pt)
    
    # calculate bounding boxes
    boxes = []
    for p in ptss:
        x1 = int(p[:, :, 0].min())
        x2 = int(p[:, :, 0].max())
        y1 = int(p[:, :, 1].min())
        y2 = int(p[:, :, 1].max())
        boxes.append([x1, y1, x2, y2])
    
    # convert to YOLO format
    yolo_boxes = []
    for box in boxes:
        yolo_box = convert_to_yolo_format(box)
        yolo_boxes.append(yolo_box)
    
    # save YOLO format annotations
    img_name = img_res[index].replace('.tif', '.txt')
    with open(os.path.join(yolo_labels_dir, img_name), 'w') as f:
        for box in yolo_boxes:
            # category ID is 0 (assuming only one category)
            line = f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
            f.write(line)

print("YOLO format annotations saved to:", yolo_labels_dir)

