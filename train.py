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
        print(f"{n_gpus} GPU's available:")
        for gpu_idx in range(n_gpus):
            print(f"\t-At device cuda:{gpu_idx} -> device model = {torch.cuda.get_device_name(gpu_idx)}")
    else:
        print(f"Cuda available with device {device} -> device model = {torch.cuda.get_device_name(device_slot)}")
    
    # Select a GPU
    device_slot = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_slot}")
else:
    n_gpus = 0
    device = torch.device("cpu")
    print(f"Cuda is not available, using {device} instead")
 

#######################
# Training parameters #
#######################

epochs = 50
batch_size = 32
# Processes for loading data in parallel
num_workers = 3
# Enables multi-gpu training if it is possible
multi_gpu = True
# Pin memory for extra speed loading batches in GPU
pin_memory = True
# Enable tensorboard
tensorboard = False

# Experiment name
exp_name = "u-net_SGD"

###################
# Data generators #
###################

# Load dataset info
data_df = pd.read_csv("../dataset/data_partition.csv")
# Create train datagen
train_dataset = Cells_dataset(data_df, "train")
train_datagen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
# Create develoment datagen
dev_dataset = Cells_dataset(data_df, "dev")
dev_datagen = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

########################
# Model initialization #
########################

# Build the model
model = U_net()
# Print model architecture
#print(model)

##################
# Training phase #
##################

# Get loss function
criterion = model.get_criterion()
# Get optimizer 
optimizer = model.get_optimizer()

# Initialization of the variables to store the results
best_loss = 99999
best_epoch = -1
train_losses, train_ious, test_losses, test_ious = [], [], [], []

# Scheduler for changing the value of the laearning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

# Prepare multi-gpu training if enabled
if multi_gpu and n_gpus > 1 :
    print("Preparing multi-gpu training...")
    model = nn.DataParallel(model)

# Move the model to the computing devices
model = model.to(device)

# Set the tensorboard writer
if tensorboard:
    tboard_writer = SummaryWriter(comment=exp_name)

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

# Plot loss and iou of training epochs
plot_results(train_losses, train_ious, test_losses, test_ious, save_as=exp_name)
