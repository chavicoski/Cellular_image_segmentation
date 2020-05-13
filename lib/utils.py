import sys
from sys import stdout
from time import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from skimage.morphology import label
from torchvision.utils import make_grid
from math import ceil

'''
Creates a png file with a grid showing the images passed as a parameter.
Params:
    images -> pytorch tensor with a batch of images
    save_as -> path of the new png file
'''
def save_images_batch(images, save_as=""):
    images_grid = make_grid(images, nrow=int(ceil(images.shape[0] / 4)))
    fig = plt.imshow(images_grid.permute(1, 2, 0))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save_as == "":
        plt.savefig("plots/images_batch.png", pad_inches=0)
    else:
        plt.savefig(save_as, pad_inches=0)

'''
Returns the rle encoding of the image.
Params:
    image -> Numpy array with the image mask. 1 = cell_mask, 0 = backgroud

Source: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
'''
def rle_encode(image):
    pixels = image.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

'''
Returns the decoded numpy array corresponding to the rle encoding.
Params:
    mask_rle -> string with the rle encoding
    shape -> target shape of the decoded numpy array

Source: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
'''
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[-2]*shape[-1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape[-2:])

'''
Returns the rle encoding for each cell in the mask.
'''
def get_cells_rle(mask, threshold=0.5):
    cells_masks = label(mask > threshold) 
    for i in range(1, cells_masks.max()+1):
        yield rle_encode(cells_masks==i)

'''
Loads the image and preprocess it in order to feed it to the network.
Params:
    image_path -> path to the image png
    target_size -> input size of the network (without channel dim)
'''
def preprocess_inference(image_path, target_size=(256, 256)):
    process_img = transforms.Compose([transforms.Resize(target_size, Image.BICUBIC), transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_img(image)
    return image_tensor.view((1,) + image_tensor.shape), image.size

'''
Metric function to compute "intersection over union".
Params:
    logits -> Tensor output of the net. Shape: (batch_size, 1, h, w)
    target -> Target mask tensor of 0's anb 1's. Shape: (batch_size, 1, h, w)
    threshold -> threshold value to count logits as 1 or 0
'''
def iou_metric(logits, target, threshold=0.5):
    # To avoid dividing by 0
    epsilon = 1e-6

    # Remove channel dimension. From (batch, channel, h, w) to (batch, h, w)
    logits = logits.squeeze(1)
    target = target.squeeze(1)

    # Cast to byte type for computation of intersection and union
    logits = (logits > threshold).byte() # Also use threshold to conver to 0 or 1
    target = target.byte()

    intersection = (logits & target).float().sum((1,2))
    union = (logits | target).float().sum((1,2))

    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou


'''
Computes the average of applying the iou with diferent mask thresholds. 
From 0.5 to 0.95 with a step of 0.05
Params:
    logits -> Tensor output of the net. Shape: (batch_size, 1, h, w)
    target -> Target mask tensor of 0's anb 1's. Shape: (batch_size, 1, h, w)
'''
def competition_iou_metric(logits, target):
    ious_sum = 0
    thresholds = np.arange(0.5, 1, 0.05)
    for threshold in thresholds:
        ious_sum += iou_metric(logits, target, threshold)
    return  ious_sum / len(thresholds) 

'''
Given the predicted masks from the net and the original masks(with different shapes).
It resizes each predicted mask to the shape of his target mask.
Params:
    out_masks -> pytorch tensor with the predicted masks
    target -> list with the target masks (with different shapes)
'''
def resize_out_masks(out_mask, target):
    resize_func = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target.shape[-2:]),
        transforms.ToTensor()
    ])
    resized_mask = resize_func(out_mask.view(out_mask.shape[-3:]))
    return resized_mask.view((1,) + resized_mask.shape)


'''
Training loop function
Params:
    train_loader -> pytorch DataLoader for training data
    net -> pytorch model
    criterion -> pytorch loss function
    optimizer -> pytorch optimizer
    device -> pytorch computing device
    pin_memory-> flag to enable pined memory to load batches into GPU
'''
def train(train_loader, net, criterion, optimizer, device, pin_memory):
    # Set the net in train mode
    net.train()

    # Initialize stats
    running_loss = 0.0
    running_iou = 0.0
    samples_count = 0
    
    # Epoch timer
    epoch_timer = time()

    # Training loop
    for batch_idx, batch in enumerate(train_loader, 0):
        # Batch timer
        batch_timer = time()
        # Get input and target from batch
        data, target = batch["image"], batch["mask"]
        # Move tensors to computing device
        data = data.to(device, non_blocking=pin_memory)
        target = target.to(device, non_blocking=pin_memory)
        # Accumulate the number of samples in the batch 
        samples_count += len(data)
        # Reset gradients
        optimizer.zero_grad()
        # Forward
        outputs = net(data)
        loss = criterion(outputs, target)
        # Backward
        loss.backward()
        optimizer.step()
        # Accumulate loss
        running_loss += loss.item()
        # Compute and Accumulate iou values
        batch_iou = iou_metric(outputs, target)
        running_iou += batch_iou.sum().item()
        # Compute current statistics
        current_loss = running_loss / samples_count 
        current_iou = running_iou / samples_count
        # Compute time per batch (in miliseconds)
        batch_time = (time() - batch_timer) * 1000
        # Print training log
        stdout.write(f"\rTrain batch {batch_idx+1}/{len(train_loader)} - {batch_time:.1f}ms/batch - loss: {current_loss:.5f} - iou: {current_iou:.5f}")

    # Compute total epoch time (in seconds)
    epoch_time = time() - epoch_timer
    # Final print with the total time of the epoch
    stdout.write(f"\rTrain batch {batch_idx+1}/{len(train_loader)} - {epoch_time:.1f}s {batch_time:.1f}ms/batch - loss: {current_loss:.5f} - iou: {current_iou:.5f}")

    # return final loss and accuracy
    return current_loss, current_iou


'''
Validation function. Computes loss and accuracy for development set.
Params:
    dev_loader -> pytorch DataLoader for development data
    net -> pytorch model
    criterion -> pytorch loss function
    device -> pytorch computing device
    pin_memory-> flag to enable pined memory to load batches into GPU
'''
def validate(dev_loader, net, criterion, device, pin_memory):
    # Set the net in eval mode
    net.eval()

    # Initialize stats
    dev_loss = 0   
    iou = 0.0

    # Validation timer
    dev_timer = time()

    # Set no_grad to avoid gradient computations
    with torch.no_grad():     
        # Testing loop
        for batch in dev_loader:       
            # Get input and target from batch
            data, target = batch["image"], batch["mask"]
            # Move tensors to computing device
            data = data.to(device, non_blocking=pin_memory)
            target = target.to(device, non_blocking=pin_memory)       
            # Compute forward and get output logits
            output = net(data)       
            # Compute loss and accumulate it
            dev_loss += criterion(output, target).item()       
            # Compute samples iou
            batch_iou = iou_metric(output, target)
            iou += batch_iou.sum().item()

    # Compute final loss
    dev_loss /= len(dev_loader.dataset)   
    # Compute final iou
    dev_iou = iou / len(dev_loader.dataset) 
    # Compute time consumed 
    dev_time = time() - dev_timer
    # Print validation log 
    stdout.write(f'\nValidation {dev_time:.1f}s: val_loss: {dev_loss:.5f} - val_iou: {dev_iou:.5f}\n')    

    return dev_loss, dev_iou


'''
Test function that computes the score like the competition. It takes the generated masks from our model
and then resizes them to fit the target shape for each of them (as not all the samples have the same shape).
Then it computes the average of all the iou scores by changing the threshold of the mask from 0.05 to 0.95 
with a step of 0.05.
Params:
    test_loader -> pytorch DataLoader for test data
    net -> pytorch model
    criterion -> pytorch loss function
    device -> pytorch computing device
    pin_memory-> flag to enable pined memory to load batches into GPU
'''
def run_competition_test(test_loader, net, device, pin_memory):
    # Set the net in eval mode
    net.eval()

    # Initialize stats
    test_loss = 0   
    iou = 0.0

    # Test timer
    test_timer = time()

    # Set no_grad to avoid gradient computations
    with torch.no_grad():     
        # Testing loop
        for batch in test_loader:       
            # Get input and target from batch
            data, target = batch["image"], batch["mask"]
            # Move the input tensor to computing device
            data = data.to(device, non_blocking=pin_memory)
            # Compute forward and get output logits
            output = net(data)       
            # Resize the output to compare iou with target mask
            resized_output = resize_out_masks(output.cpu(), target)
            # Compute samples iou
            batch_iou = competition_iou_metric(resized_output, target)
            iou += batch_iou.sum().item()

    # Compute final iou
    test_iou = iou / len(test_loader.dataset) 
    # Compute time consumed 
    test_time = time() - test_timer
    # Print validation log 
    stdout.write(f'\nTest {test_time:.1f}s: averaged_iou_score = {test_iou:.5f}\n')    

    return test_iou


'''
Auxiliary funtion to make the plots for loss and iou metric from training phase.
Params:
    train_losses -> list with the loss value of train split for each epoch
    train_ious -> list with the iou value for train split for each epoch
    val_losses -> list with the loss value of development split for each epoch
    val_ious -> list with the iou value for development split for each epoch
    loss_title -> String with the title of the losses plot
    iou_title -> String with the title of the ious plot
    save_as -> String with the name of the png file to save the plot. Default is set to not save the plot
'''
def plot_results(train_losses, train_ious, dev_losses, dev_ious, loss_title="Loss", iou_title="Intersection Over Union", save_as=""):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
    ax[0].plot(train_losses, "r", label="train")
    ax[0].plot(dev_losses, "g", label="dev")
    ax[0].legend()
    ax[0].title.set_text(loss_title)
    ax[1].plot(train_ious, "r", label="train")
    ax[1].plot(dev_ious, "g", label="dev")
    ax[1].legend()
    ax[1].title.set_text(iou_title)
    if save_as is not "": 
        plt.savefig("plots/" + save_as + "_trainres.png")
    else:
        plt.show() 


if __name__ == "__main__":

    ##################
    # Test functions #
    ##################

    dummy_mask = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [1, 0, 1, 1 ,1], [1, 0, 0, 1, 1]])
    print(f"\nDummy mask orig:\n{dummy_mask}")
    print(f"\nDummy mask orig T:\n{dummy_mask.T}")
    rles = list(get_cells_rle(dummy_mask))
    print(f"\nCells rles from dummy mask: {rles}")
    print(f"\nSingle cells masks reconstruction:")
    for rle in rles:
        print(rle_decode(rle, dummy_mask.shape), "\n")
