import sys
sys.path.insert(1, ".")
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from lib.utils import rle_decode

'''
This script takes the test set from the stage one of the competition and a csv with the rle encodings
of the masks to generate a csv to access the data in an easy way and process the masks in order to check
our predictions with them.
'''

# Data paths
dataset_path = "../dataset"
test_path = os.path.join(dataset_path, "test1")
test_labels_path = os.path.join(dataset_path, "test1_labels.csv")

# Original dataframe
df = pd.read_csv(test_labels_path)
# New dataframe
new_df = pd.DataFrame(df["ImageId"].unique(), columns=["ImageId"])
# Set partition name
new_df["Partition"] = "test"

##################################
# Images and Masks preprocessing #
##################################

# Target size to reshape the images
target_size=(256, 256)

# Transformations for images -> 1. Resize, 2. Normalize ([0, 1]) and create tensor with color channel
process_img = transforms.Compose([transforms.Resize(target_size, Image.BICUBIC), transforms.ToTensor()])

# Add the columns for the masks shapes to the new dataframe
new_df["mask_height"] = ""
new_df["mask_width"] = ""
for index, row in new_df.iterrows():
    aux_row = df[df["ImageId"] == row["ImageId"]].iloc[0]
    row["mask_height"] = aux_row["Height"]
    row["mask_width"] = aux_row["Width"]

# Initialize the columns to store the paths to the preprocessed images and masks
new_df["ImagePath"] = ""
new_df["MaskPath"] = ""

# Preprocessing loop
for index, row in tqdm(new_df.iterrows()):
    # Get the id of the sample
    image_id = row["ImageId"]
    # Open the input image and transform from RGBA to RGB
    img = Image.open(os.path.join(test_path, image_id, "images", f"{image_id}.png")).convert("RGB")
    # Apply processing to image and get a pytorch tensor
    img_tensor = process_img(img)

    # Build the union of the masks
    mask_shape = (row["mask_height"], row["mask_width"])
    mask_tensor = torch.zeros((1,) + mask_shape)
    for i, mask_data in df[df["ImageId"] == image_id].iterrows():
        # Decode the rle mask to get a numpy tensor
        aux_mask = rle_decode(mask_data["EncodedPixels"], mask_shape)
        # Add channel dimension
        aux_mask = aux_mask.reshape((1,) + mask_shape)
        # Convert the numpy to pytorch tensor
        aux_mask_tensor = torch.from_numpy(aux_mask)
        # Accumulate the mask with the others
        mask_tensor += aux_mask_tensor

    # Clamp values to between 0 and 1
    mask_tensor = torch.clamp(mask_tensor, min=0, max=1)

    # Store the image and mask tensors for training
    img_tensor_path = os.path.join(test_path, image_id, "images", f"{image_id}.pt")
    mask_tensor_path = os.path.join(test_path, image_id, "images", f"masks_union.pt")
    torch.save(img_tensor, img_tensor_path)
    torch.save(mask_tensor, mask_tensor_path)
    # Save the tensor's paths in the dataframe
    row["ImagePath"] = img_tensor_path
    row["MaskPath"] = mask_tensor_path
    
# Save the new dataframe
new_df.to_csv(os.path.join(dataset_path, "test1_partition.csv"))

