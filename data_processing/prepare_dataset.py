import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

'''
This script takes the first available partition of the dataset from the stage 1 of the competition
and makes the train/dev split using a csv and processes all the images and masks to have the desired
shape and format to train(converts the png files to pytorch tensors).
'''

# Data paths
dataset_path = "../dataset"
train_path = os.path.join(dataset_path, "train")
train_labels_path = os.path.join(dataset_path, "train_labels.csv")

# Original dataframe
df = pd.read_csv(train_labels_path)
# New dataframe to store usefull train data
new_df = pd.DataFrame(df["ImageId"].unique(), columns=["ImageId"])

########################################
# DATASET PARTITIONING (train and dev) #
########################################

# Set the percentaje of samples for the development partition
dev_partition = 0.2

# Add the partition column and initialize it to 'train'
new_df["Partition"] = "train"
# Get number of samples
n_samples = len(new_df.index)
# Compute number of samples for development partition
n_samples_dev = int(n_samples * dev_partition)
# Change the partition asignation in the dataframe
new_df["Partition"].iloc[:n_samples_dev] = "dev"

##################################
# Images and Masks preprocessing #
##################################
'''
Each cell mask is in a separated .png, so we are going to generate the union
of this masks to avoid making this computation during training.

The masks generated and the input images are going to be preprocessed to store them
in a pytorch tensor with the size and format ready for training to avoid this computation
in the input pipeline.
'''
# Target size to reshape the images and masks
target_size=(256, 256)

# Transformations for images -> 1. Resize, 2. Normalize ([0, 1]) and create tensor with color channel
process_img = transforms.Compose([transforms.Resize(target_size, Image.BICUBIC), transforms.ToTensor()])

# Initialize the columns to store the paths to the preprocessed images and masks
new_df["ImagePath"] = ""
new_df["MaskPath"] = ""

# Preprocessing loop
for index, row in tqdm(new_df.iterrows()):
    
    # Get the id of the sample
    image_id = row["ImageId"]
    # Open the input image and transform from RGBA to RGB
    img = Image.open(os.path.join(train_path, image_id, "images", f"{image_id}.png")).convert("RGB")
    # Apply processing to image and get a pytorch tensor
    img_tensor = process_img(img)

    # Build the union of the masks
    mask_tensor = torch.zeros((1,)+target_size)
    for mask_f in os.listdir(os.path.join(train_path, image_id, "masks")):
        if mask_f.endswith(".png"):
            # Load the mask
            aux_mask = Image.open(os.path.join(train_path, image_id, "masks", mask_f))
            # Apply processing to mask and get a pytorch tensor
            aux_mask_tensor = process_img(aux_mask)
            # Accumulate the mask with the others
            mask_tensor += aux_mask_tensor

    # Clamp values to between 0 and 1
    mask_tensor = torch.clamp(mask_tensor, min=0, max=1)

    # Store the image and mask tensors for training
    img_tensor_path = os.path.join(train_path, image_id, "images", f"{image_id}.pt")
    mask_tensor_path = os.path.join(train_path, image_id, "masks", f"masks_union.pt")
    torch.save(img_tensor, img_tensor_path)
    torch.save(mask_tensor, mask_tensor_path)
    # Save the tensor's paths in the dataframe
    row["ImagePath"] = img_tensor_path
    row["MaskPath"] = mask_tensor_path
    

# Save the new dataframe
new_df.to_csv(os.path.join(dataset_path, "data_partition.csv"))

