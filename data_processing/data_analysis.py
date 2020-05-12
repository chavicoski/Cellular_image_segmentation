import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

# Data paths
dataset_path = "../dataset"
train_path = os.path.join(dataset_path, "train")
train_labels_path = os.path.join(dataset_path, "train_labels.csv")

# Load dataframe
df = pd.read_csv(train_labels_path)

# Stats initialization
shapes_dict = dict()
max_intensities = []
min_intensities = []
empty_masks = 0
total_samples = 0

# Preprocessing loop
for image_id in tqdm(df["ImageId"].unique()):
    # Open the input image and transform from RGBA to RGB
    img = Image.open(os.path.join(train_path, image_id, "images", f"{image_id}.png")).convert("RGB")
    img_arr = np.asarray(img)
    shapes_dict[str(img_arr.shape)] = shapes_dict.get(str(img_arr.shape), 0) + 1
    max_intensities.append(img_arr.max())
    min_intensities.append(img_arr.min())

    is_empty = True
    for mask_f in os.listdir(os.path.join(train_path, image_id, "masks")):
        if mask_f.endswith(".png"):
            # Load the mask
            aux_mask = np.asarray(Image.open(os.path.join(train_path, image_id, "masks", mask_f)))
            if aux_mask.max() > 0:
                is_empty = False
    
    if is_empty: empty_masks += 1
    total_samples += 1


# Print stats
print(f"Total number of samples: {total_samples}")
print(f"Empty masks: {empty_masks}")
for shape, count in sorted(shapes_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"{shape:<15}: {count:>3}")

# Histogram version of slices counts
plt.hist(max_intensities, bins=30, label="max pixel")
plt.hist(min_intensities, bins=30, label="min pixel")
plt.legend(loc="upper right")
plt.xlabel("Pixel intensity")
plt.ylabel("Count")
plt.title(f"Count of max and min pixels for each sample")
plt.savefig(f"plots/pixels_values_count.png")
plt.clf()  # Reset figure for next plot
