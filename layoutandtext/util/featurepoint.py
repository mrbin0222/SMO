import cv2
import numpy as np


# extract edge contours
def edge(image, seed=24):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # morphology operation, open operation kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # open operation
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    # median filter
    opened = cv2.medianBlur(opened,3)
    # edge summary
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    t = []
    for i,p in enumerate(contours):
        area = cv2.contourArea(p)
        if area >= seed:
            t.append(p)
    
    # sort by edge significance
    t=sorted(t, key=lambda x: len(x), reverse=True)

    return t

# get edge point coordinate mapping map
def get_bz(s, image):
    # get edge point set, coordinates are w,h
    p_r = np.concatenate(s).astype(int).squeeze()
    # create bool type empty mapping map, h,w
    bz = np.zeros((image.shape[0],image.shape[1]), dtype=bool)
    # assign values to the mapping map according to all coordinates
    for p in p_r:
        bz[p[1]][p[0]] = True
    return bz