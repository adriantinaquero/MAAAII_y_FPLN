from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import Ship


def load_dataset(route: str) -> tuple:

    transform_basic = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_aug = transforms.Compose([
        transforms.Resize(64),
        transforms.Pad(4),
        transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.75,1.25), shear=15),
        transforms.ColorJitter(brightness=(0.2,0.8), contrast=(0.2, 0.8)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_basic = Ship(route, transform_basic)
    train_aug = Ship(route, transform_aug)
    test_dataset = Ship(route, transform_basic)

    batch_size = 128

    train_loader_basic = DataLoader(train_basic, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader_aug = DataLoader(train_aug, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return (train_loader_basic, train_loader_aug, test_loader)