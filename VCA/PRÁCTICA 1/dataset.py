import torch.utils.data as data
from torchvision.io import decode_image
import torchvision.transforms as transforms
import os

class ShipData(data.Dataset):
    
    def __init__(self, csv_path, transform = None):
        super().__init__()
        csv_path = csv_path.replace("\\", "/")
        img_path = "/".join(csv_path.split("/")[:-1]) + "/images/"
        with open(csv_path) as csv_file:
            self.img_files = [os.path.join(img_path, i).strip("\n").split(";") for i in csv_file.readlines()[1:]]

        if transform:
          self.transform = transform
        else:
          self.transform = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.ToTensor()])


    def __getitem__(self, index):
        img_path, label = self.img_files[index]

        image = decode_image(img_path)
        image = self.transform(image)

        return image, int(label)

    def __len__(self):
        return len(self.img_files)

if __name__ == "__main__":
   a = ShipData("dataset/ship.csv")
   print(len(a))
   a[0]