import torch,tqdm
from torch.utils.data import DataLoader,random_split
from data_set import CustomImageDataset
from models import Visual_encoder
from parameters import batch_size,device,encoder
from helper import pad_str,encoding
import matplotlib.pyplot as plt
from text_style import TextEncoder_FC
from  block import AdaLN,Generator_Resnet
from temp2 import GenModel_FC
def show_images(source, target):
    # Reshape the tensors to image shape (height, width, channels)
        source = source.detach().numpy()
        target = target.detach().numpy()

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the source image in the first subplot
        axs[0].imshow(source)
        axs[0].axis('off')
        axs[0].set_title('Source Image')

        # Plot the target image in the second subplot
        axs[1].imshow(target)
        axs[1].axis('off')
        axs[1].set_title('Target Image')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Display the figure
        plt.show()
if __name__ == "__main__":

    TextDatasetObj = CustomImageDataset()
    train_ratio = 0.8
    test_ratio = 1 - train_ratio

    # Calculate the sizes of train and test sets based on the split ratios
    train_size = int(train_ratio * len(TextDatasetObj))
    test_size = len(TextDatasetObj) - train_size

    # Split the dataset into train and test sets
    train_set, test_set = random_split(TextDatasetObj, [train_size, test_size])

    # Define batch size and number of workers for DataLoader
    num_workers = 1
    

# Example usage

    # Create DataLoader instances for train and test sets
    #train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    #test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    # visual_net=Visual_encoder().to(device=device)

    batch_size = 2

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    for batch_idx, (images, labels) in enumerate(train_loader):
            # The __getitem__ function will be called twice for each iteration
            # The data for each item will be stored internally and organized into batches

            print(f"Iteration {batch_idx}: {len(images)}, {len(labels)}")
