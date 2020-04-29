import torch
from torchvision.transforms import functional as transformsF
from random import random, randint, uniform

class Cells_dataset(torch.utils.data.Dataset):
    '''
    Dataset constructor

    Params:
        data_df -> Pandas dataframe with the dataset info
            *data_df columns:
                - ImageId -> sample id in the dataset
                - Partition -> 'train' or 'dev'
                - ImagePath -> path to input tensor
                - MaskPath -> path to mask output tensor

        partition -> Select the dataset partition of the generator. Can be "train" or "dev"
	data_augmentation -> Flag to enable data augmentation
    '''
    def __init__(self, data_df, partition="train", data_augmentation=False, make_crops=False):
        
        self.partition = partition
        # Store the samples from the selected partition
        self.df = data_df[data_df["Partition"]==partition]
        self.data_augmentation = data_augmentation
        self.make_crops = make_crops

    '''
    Returns the number of samples in the dataset
    '''
    def __len__(self):
        # Returns the number of rows in the dataframe
        return len(self.df.index)

    '''
    Generates a sample of data -> (input_image, output_mask)
    '''
    def __getitem__(self, index):

        # Get the dataframe row of the sample
        sample_row = self.df.iloc[index]
        # Load the data
        image_tensor = torch.load(sample_row["ImagePath"])
        mask_tensor = torch.load(sample_row["MaskPath"])
        # Apply data augmentation (if enabled)
        if self.data_augmentation:
            image_tensor, mask_tensor = self.apply_data_augmentation(image_tensor, mask_tensor)
       
        return {"image": image_tensor, "mask": mask_tensor}

    '''
    Applies data augmentation to a given pytorch tensors (x:sample and y:label)
    '''
    def apply_data_augmentation(self, x_tensor, y_tensor):

        c, h, w = x_tensor.shape

        # From pytorch tensor to PIL image (to apply transforms)
        x_image = transformsF.to_pil_image(x_tensor)
        y_image = transformsF.to_pil_image(y_tensor)

        # Random horizontal flip
        if random() > 0.5:
            x_image = transformsF.hflip(x_image)
            y_image = transformsF.hflip(y_image)

        # Random vertical flip
        if random() > 0.5:
            x_image = transformsF.vflip(x_image)
            y_image = transformsF.vflip(y_image)

        # Compute random affine transformation parameters
        rot_angle = uniform(-10, 10)
        translate = (randint(0, int(h*0.1)), randint(0, int(w*0.1)))
        scale = uniform(0.9, 1.1)
        shear = uniform(-10, 10)
        # Apply affine transformation
        x_image_trans = transformsF.affine(x_image, rot_angle, translate, scale, shear)
        y_image_trans = transformsF.affine(y_image, rot_angle, translate, scale, shear)

        # Make random crop
        if self.make_crops:
            # Coordiantes of the upper left corner of the crop
            vertical_coord = randint(0, int(h * 0.2))  
            horizontal_coord = randint(0, int(w * 0.2))  
            # Size of the crop
            crop_height = int(h * 0.8)
            crop_width = int(w * 0.8)
            # Target size to resize the crop
            target_size = [h, w]
            # Make the same crop in the image and the mask
            x_image_trans = transformsF.resized_crop(x_image_trans, vertical_coord, horizontal_coord, crop_height, crop_width, target_size) 
            y_image_trans = transformsF.resized_crop(y_image_trans, vertical_coord, horizontal_coord, crop_height, crop_width, target_size) 

        # From PIL image to pytorch tensor again
        x_tensor_trans = transformsF.to_tensor(x_image_trans)
        y_tensor_trans = transformsF.to_tensor(y_image_trans)

        return x_tensor_trans, y_tensor_trans


			
if __name__ == "__main__":
	
    import pandas as pd
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    print("TEST DATA GENERATOR")

    data_df = pd.read_csv("../dataset/data_partition.csv")
    dataset = Cells_dataset(data_df, "train", data_augmentation=True, make_crops=True)
    dataloader = DataLoader(dataset, batch_size = 8)
    batch = next(iter(dataloader))
     
    images_grid = make_grid(batch["image"], nrow=4)
    masks_grid = make_grid(batch["mask"], nrow=4)

    plt.imshow(images_grid.permute(1, 2, 0))
    plt.savefig("plots/test_dataloader_images.png")
    plt.clf()

    plt.imshow(masks_grid.permute(1, 2, 0))
    plt.savefig("plots/test_dataloader_masks.png")
    
