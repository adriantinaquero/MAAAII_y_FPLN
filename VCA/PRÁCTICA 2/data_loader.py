import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import OCTDataset


def load_dataset(image_route: str, mask_route: str, batch_size, train_size=0.7, val_size=0.15) -> tuple:

    transform_basic = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((208, 312)),
        transforms.ToTensor(),
    ])
    transform_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((208, 312)),
        transforms.CenterCrop((208, 208)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])

    dataset_basic = OCTDataset(image_path=image_route, mask_path=mask_route, transform=transform_basic)
    dataset_aug = OCTDataset(image_path=image_route, mask_path=mask_route, transform=transform_aug)
    dataset_size = len(dataset_basic)

    train_size = int(train_size * dataset_size)
    val_size = int(val_size * dataset_size)
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
    a = load_dataset("VCA/PRÁCTICA 2/dataset/images", "VCA/PRÁCTICA 2/dataset/masks", 128)
    for i, p in a[1]:
        for j, k in zip(i, p):
            fig, ax = plt.subplots(1,2, figsize=(8, 4))
            ax[0].imshow(j.permute(1, 2, 0).numpy(), vmin=0, vmax=1, cmap="gray")
            ax[1].imshow(k.permute(1, 2, 0).numpy(), cmap="gray")
            plt.tight_layout()
            plt.show()
