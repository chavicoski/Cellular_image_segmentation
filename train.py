import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from matplotlib import pyplot as plt
from lib.data_generators import Cells_dataset
from lib.utils import *
from models.my_models import U_net

# Check computing device
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
 

#######################
# Training parameters #
#######################

epochs = 300
batch_size = 32

# Optimizer config
optimizer_name = "Adam"  # Options "Adam", "SGD"
learning_rate = 0.001

# Weight initializer of the model 
initializer = "he_normal"  # Options "he_normal", "dirac", "xavier_uniform", "xavier_normal"

# Model settings
use_batchnorm = True
dropout = 0.5  # Dropout before the upsampling part

# Data loader settings
data_augmentation = True
make_crops = True  # For making random crops after the data agumentation
num_workers = 2    # Processes for loading data in parallel
multi_gpu = True   # Enables multi-gpu training if it is possible
pin_memory = True  # Pin memory for extra speed loading batches in GPU

# Enable tensorboard
tensorboard = True

# Experiment name for saving logs
exp_name = f"u-net_{optimizer_name}-{learning_rate}_{initializer}_{dropout}-dropout"
if use_batchnorm: exp_name += "_batchnorm"
if data_augmentation: exp_name += "_DA"
if make_crops: exp_name += "_crops"

print(f"\nGoing to run experiment {exp_name}")

###################
# Data generators #
###################

# Load dataset info
data_df = pd.read_csv("../dataset/data_partition.csv")
# Create train datagen
train_dataset = Cells_dataset(data_df, "train", data_augmentation=data_augmentation, make_crops=make_crops)
train_datagen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
# Create develoment datagen
dev_dataset = Cells_dataset(data_df, "dev")
dev_datagen = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

########################
# Model initialization #
########################

# Build the model
model = U_net(batch_norm=use_batchnorm, dropout=dropout)
# Initialize the weights
model.init_weights(initializer)
# Print model architecture
print(f"\nModel topology:\n{model}")

##################
# Training phase #
##################

# Get loss function
criterion = model.get_criterion()
# Get optimizer 
optimizer = model.get_optimizer(opt=optimizer_name, lr=learning_rate)
print(f"\nGoing to train with {optimizer_name} with lr={learning_rate}")

# Initialization of the variables to store the results
best_loss = 99999
best_epoch = -1
train_losses, train_ious, test_losses, test_ious = [], [], [], []

# Scheduler for changing the value of the laearning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

# Set the tensorboard writer
if tensorboard:
    tboard_writer = SummaryWriter(comment=exp_name)

# Prepare multi-gpu training if enabled
if multi_gpu and n_gpus > 1 :
    print("\nPreparing multi-gpu training...")
    model = nn.DataParallel(model)

# Move the model to the computing devices
model = model.to(device)

# Print training header
print("\n############################\n"\
      +f"# TRAIN PHASE: {epochs:>4} epochs #\n"\
      + "############################\n")

# Start training
for epoch in range(epochs):
    # Header of the epoch log
    stdout.write(f"Epoch {epoch}: ") 
    if best_epoch > -1 : stdout.write(f"current best loss = {best_loss:.5f}, at epoch {best_epoch}\n")
    else: stdout.write("\n")
    # Train split
    train_loss, train_iou = train(train_datagen, model, criterion, optimizer, device, pin_memory)
    # Development split 
    test_loss, test_iou = test(dev_datagen, model, criterion, device, pin_memory)
    # Apply the lr scheduler
    scheduler.step(test_loss)
    # Save the results of the epoch
    train_losses.append(train_loss), train_ious.append(train_iou) 
    test_losses.append(test_loss), test_ious.append(test_iou)
    # Log in tensorboard 
    if tensorboard:
        # Loss
        tboard_writer.add_scalar("Loss/train", train_loss, epoch)
        tboard_writer.add_scalar("Loss/test", test_loss, epoch)
        # Intersection over union
        tboard_writer.add_scalar("IoU/train", train_iou, epoch)
        tboard_writer.add_scalar("IoU/test", test_iou, epoch)

    # If val_loss improves we store the model
    if test_losses[-1] < best_loss:
        model_path = f"models/checkpoints/{exp_name}_best"
        print(f"Saving new best model in {model_path}")
        # Save the entire model
        torch.save(model, model_path)
        # Update best model stats
        best_loss = test_losses[-1]
        best_epoch = epoch

    # To separate the epochs outputs  
    stdout.write("\n")

# Add tensorboard entry with the experiment result
if tensorboard:
	tboard_writer.add_hparams({"optimizer": optimizer_name,
                    "lr": learning_rate,
                    "initializer": initializer,
                    "batch_norm": use_batchnorm,
                    "dropout": dropout,
                    "data_augmentation": data_augmentation,
                    "crops": make_crops},
                    {"hparam/loss": best_loss, "hparam/iou": test_ious[best_epoch], "hparam/best_epoch": best_epoch})

# Close the tensorboard writer
if tensorboard:
    tboard_writer.close()

# Plot loss and iou of training epochs
plot_results(train_losses, train_ious, test_losses, test_ious, save_as=exp_name)
