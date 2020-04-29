from sys import stdout
from time import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.morphology import label

'''
Returns the rle encoding of the image.
Params:
    image -> Numpy array with the image mask. 1 = cell_mask, 0 = backgroud

Source: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
'''
def rle_encode(image):
    pixels = image.flatten()
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
Training loop function
Params:
    train_loader -> pytorch DataLoader for training data
    net -> pytorch model
    criterion -> pytorch loss function
    optimizer -> pytorch optimizer
    device -> pytorch computing device
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
Test function. Computes loss and accuracy for development set.
Params:
    test_loader -> pytorch DataLoader for testing data
    net -> pytorch model
    criterion -> pytorch loss function
    device -> pytorch computing device
'''
def test(test_loader, net, criterion, device, pin_memory):
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
            # Move tensors to computing device
            data = data.to(device, non_blocking=pin_memory)
            target = target.to(device, non_blocking=pin_memory)       
            # Compute forward and get output logits
            output = net(data)       
            # Compute loss and accumulate it
            test_loss += criterion(output, target).item()       
            # Compute samples iou
            batch_iou = iou_metric(output, target)
            iou += batch_iou.sum().item()


    # Compute final loss
    test_loss /= len(test_loader.dataset)   
    # Compute final iou
    test_iou = iou / len(test_loader.dataset) 
    # Compute time consumed 
    test_time = time() - test_timer
    # Print test log 
    stdout.write(f'\nTest {test_time:.1f}s: val_loss: {test_loss:.5f} - val_iou: {test_iou:.5f}\n')    

    return test_loss, test_iou


'''
Auxiliary funtion to make the plots for loss and iou metric from training phase.
Params:
    train_losses -> list with the loss value of train split for each epoch
    train_ious -> list with the iou value for train split for each epoch
    test_losses -> list with the loss value of test split for each epoch
    test_ious -> list with the iou value for test split for each epoch
    loss_title -> String with the title of the losses plot
    iou_title -> String with the title of the ious plot
    save_as -> String with the name of the png file to save the plot. Default is set to not save the plot
'''
def plot_results(train_losses, train_ious, test_losses, test_ious, loss_title="Loss", iou_title="Intersection Over Union", save_as=""):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
    ax[0].plot(train_losses, "r", label="train")
    ax[0].plot(test_losses, "g", label="test")
    ax[0].legend()
    ax[0].title.set_text(loss_title)
    ax[1].plot(train_ious, "r", label="train")
    ax[1].plot(test_ious, "g", label="test")
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
    dummy_mask = dummy_mask.reshape((1,) + dummy_mask.shape)  # Add channel dimension
    print(f"\nDummy mask orig:\n{dummy_mask}")
    rles = list(get_cells_rle(dummy_mask))
    print(f"\nCells rles from dummy mask: {rles}")
    print(f"\nSingle cells masks reconstruction:")
    for rle in rles:
        print(rle_decode(rle, dummy_mask.shape), "\n")
