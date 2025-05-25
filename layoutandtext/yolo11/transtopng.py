import os

import cv2
import numpy as np
from tqdm import tqdm


def convert_tif_to_png():
    # convert tif image to png format
    base_dir = "MoNuSegTrainingData"
    
    # process training set and validation set
    for split in ['test']:
        print(f"\nprocess {split} set...")
        img_dir = os.path.join(base_dir, split, 'images')
        
        # get all tif files
        tif_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        
        # use tqdm to show progress
        for img_file in tqdm(tif_files, desc=f"convert {split} set images"):
            # build complete file path
            tif_path = os.path.join(img_dir, img_file)
            png_path = tif_path.replace('.tif', '.png')
            
            try:
                # read tif image
                img = cv2.imread(tif_path, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"警告: 无法读取图像 {tif_path}")
                    continue
                
                # save as png format
                cv2.imwrite(png_path, img)
                
                # delete original tif file
                os.remove(tif_path)
                
                # update corresponding label file name
                label_file = img_file.replace('.tif', '.txt')
                old_label_path = os.path.join(base_dir, split, 'labels', label_file)
                new_label_path = old_label_path.replace('.tif', '.png')
                if os.path.exists(old_label_path):
                    os.rename(old_label_path, new_label_path)
                    
            except Exception as e:
                print(f"error processing {img_file}: {str(e)}")
                continue

    # update image path in data.yaml
    data_yaml_path = os.path.join(base_dir, "data.yaml")
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            content = f.read()
        content = content.replace('.tif', '.png')
        with open(data_yaml_path, 'w') as f:
            f.write(content)
        print("\nupdated image path in data.yaml")

if __name__ == "__main__":
    print("start converting image format...")
    convert_tif_to_png()
    print("\nconversion completed!")
