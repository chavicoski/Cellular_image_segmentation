import torch

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
    '''
    def __init__(self, data_df, partition="train"):
        
        self.partition = partition
        # Store the samples from the selected partition
        self.df = data_df[data_df["Partition"]==partition]

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
        # Load the image tensor
        image_tensor = torch.load(sample_row["ImagePath"])
        # Load the mask tensor
        mask_tensor = torch.load(sample_row["MaskPath"])
       
        return {"image": image_tensor, "mask": mask_tensor}
