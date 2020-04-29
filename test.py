import sys
import torch

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

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <PATH_TO_TRAINED_MODEL>")
    sys.exit()

model_path = sys.argv[1]

# Data loader settings
batch_size = 32
num_workers = 2    # Processes for loading data in parallel
multi_gpu = True   # Enables multi-gpu training if it is possible
pin_memory = True  # Pin memory for extra speed loading batches in GPU

###################
# Data generators #
###################

# Load dataset info
data_df = pd.read_csv("../dataset/data_partition.csv")
# Create develoment datagen
dev_dataset = Cells_dataset(data_df, "dev")
dev_datagen = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

#############################
# Load the pretrained model #
#############################

torch.load(model_path)
