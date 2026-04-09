import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import Ship


def load_dataset(route: str, batch_size) -> tuple:

    transform_basic = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,
          std=0.225)
    ])

    transform_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Pad(4),
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.75,1.25), shear=15),
        transforms.ColorJitter(brightness=(0.2,0.8), contrast=(0.2, 0.8)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,
          std=0.225),
    ])

    dataset_basic = Ship(route, transform_basic)
    dataset_aug = Ship(route, transform_aug)
    dataset_size = len(dataset_basic)

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - (train_size + val_size)

    train_basic, val_data, test_data = random_split(
        dataset_basic,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_aug, _, _ = random_split(
        dataset_aug,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader_basic = DataLoader(train_basic, batch_size=batch_size, shuffle=True)
    train_loader_aug = DataLoader(train_aug, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader_basic, train_loader_aug, val_loader, test_loader

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    a = load_dataset("VCA/PRÁCTICA 1/dataset/ship.csv", 180)
    for i, p in a[0]:
        for j, k in zip(i, p):
            plt.imshow(j.permute(1, 2, 0).numpy(), vmin=0, vmax=1)       # permute para pasar de [C, H, W] a [H, W, C] 
            plt.title(f"Predicción:{k}")
            plt.show()
