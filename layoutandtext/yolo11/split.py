import os
import random
import shutil
from pathlib import Path

# set random seed to ensure reproducibility
random.seed(42)

# define path
base_dir = "MoNuSegTrainingData"
images_dir = os.path.join(base_dir, "TissueImages")
labels_dir = os.path.join(base_dir, "yololabel")

# create training and validation set directories
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "val")

# create training and validation set image and label directories
train_images_dir = os.path.join(train_dir, "images")
train_labels_dir = os.path.join(train_dir, "labels")
test_images_dir = os.path.join(test_dir, "images")
test_labels_dir = os.path.join(test_dir, "labels")

# create directories
for dir_path in [train_images_dir, train_labels_dir, test_images_dir, test_labels_dir]:
    os.makedirs(dir_path, exist_ok=True)

# get all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
image_files.sort()  # ensure consistent order

# calculate test set size (20%)
test_size = int(len(image_files) * 0.2)

# randomly select test set
test_files = random.sample(image_files, test_size)
train_files = [f for f in image_files if f not in test_files]

def copy_files(file_list, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    """copy images and corresponding label files"""
    for img_file in file_list:
        # copy image
        src_img_path = os.path.join(src_img_dir, img_file)
        dst_img_path = os.path.join(dst_img_dir, img_file)
        shutil.copy2(src_img_path, dst_img_path)
        
        # copy corresponding label file
        label_file = img_file.replace('.tif', '.txt')
        src_label_path = os.path.join(src_label_dir, label_file)
        dst_label_path = os.path.join(dst_label_dir, label_file)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)

# copy training set files
print("copying training set files...")
copy_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)

# copy test set files
print("copying test set files...")
copy_files(test_files, images_dir, labels_dir, test_images_dir, test_labels_dir)

# print dataset statistics
print("\ndataset split completed!")
print(f"total image number: {len(image_files)}")
print(f"training set number: {len(train_files)}")
print(f"test set number: {len(test_files)}")

# create dataset configuration file
data_yaml = f"""path: {os.path.abspath(base_dir)}  # dataset root directory
train: train/images  # training set image relative path
val: val/images  # validation set image relative path

# categories
names:
  0: nucleus  # nucleus category
"""

# save configuration file
with open(os.path.join(base_dir, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("\ndataset configuration file generated: data.yaml")
