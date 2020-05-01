import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lib.data_generators import Cells_dataset
from lib.utils import run_competition_test

# Check script arguments
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <PATH_TO_TRAINED_MODEL>")
    sys.exit()

##########################
# Check computing device #
##########################

if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"\n{n_gpus} GPU's available:")
        for gpu_idx in range(n_gpus):
            print(f"\t-At device cuda:{gpu_idx} -> device model = {torch.cuda.get_device_name(gpu_idx)}")
    else:
        print(f"\nCuda available with device {device} -> device model = {torch.cuda.get_device_name(device_slot)}")
    
    # Select a GPU
    device_slot = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_slot}")
else:
    n_gpus = 0
    device = torch.device("cpu")
    print(f"\nCuda is not available, using {device} instead")
 

###################
# Test parameters #
###################

model_path = sys.argv[1]

# Data loader settings
batch_size = 1     # THIS MUST BE 1 IN ORDER TO WORK
num_workers = 2    # Processes for loading data in parallel
multi_gpu = True   # Enables multi-gpu training if it is possible
pin_memory = True  # Pin memory for extra speed loading batches in GPU

###################
# Data generators #
###################

# Load dataset info
data_df = pd.read_csv("../dataset/data_partition.csv")
# Create test datagen
test_dataset = Cells_dataset(data_df, "dev")
test_datagen = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

#############################
# Load the pretrained model #
#############################

model = torch.load(model_path)

########################
# Run competition test #
########################

test_iou = run_competition_test(test_datagen, model, device, pin_memory)
