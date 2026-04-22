import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms
import glob
import os
import numpy as np
import random
import matplotlib.pyplot as plt

class OCTDataset(Dataset):
    
    def __init__(self, image_path, mask_path, rsize = (416,624), transform = None):
        super().__init__()
        # Load all the filenames with extension tif from the image_path directory
        self.img_files = glob.glob(os.path.join(image_path,'*.jpg'))

        self.mask_files = []

        # We asume that each image has the same filename as its corresponding mask
        # but it is stored in another directory (mask_path)
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(mask_path, os.path.basename(img_path)))
                
        self.rsize = rsize  # Size to use in default Resize transform
        self.transform = transform

    # Returns both the image and the mask
    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = plt.imread(img_path)
        mask = plt.imread(mask_path)
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        if len(image.shape) > 2:
            image = image[:,:,0]
        mask = (mask > 100).astype(np.uint8) # Make sure that mask is binary
        # Apply the defined transformations to both image and mask
        if self.transform is not None:
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            random.seed(seed) # apply this seed to image transforms
            torch.manual_seed(seed) 
            image = self.transform(image)
            random.seed(seed) # apply the same seed to mask transforms
            torch.manual_seed(seed) 
            mask = self.transform(mask)
        else:
            t = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.rsize, interpolation= InterpolationMode.NEAREST),
                transforms.ToTensor()])

            image = t(image)
            mask = t(mask)
        
        return image, mask

    def __len__(self):
        return len(self.img_files)
    
    def show(self, image, mask, title=None):
        fig, ax = plt.subplots(1,2, figsize=(8, 4))
        ax[0].imshow(image, cmap="gray")
        ax[0].axis('off')
        if title is not None:
            fig.suptitle(title)
        ax[1].imshow(mask, cmap="gray")
        ax[1].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    route = "VCA/PRÁCTICA 2"
    simple_dataset = OCTDataset(f"{route}/dataset/images", f"{route}/dataset/masks")
    print("Dataset len:", len(simple_dataset))
    nsamples = 1
    for _ in range(nsamples):
        idx = np.random.randint(0, len(simple_dataset))
        im, mask = simple_dataset[idx]
        simple_dataset.show(im.squeeze(), mask.squeeze(), title=f"Sample {idx}")