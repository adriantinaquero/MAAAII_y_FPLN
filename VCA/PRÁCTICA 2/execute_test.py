import sys
import torch
from dataset import OCTDataset
from evaluate import evaluate_model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if len(sys.argv) != 4:
        print("Usage: python execute_test.py modelpath datasetpath")
        modelpath = "VCA/PRÁCTICA 2/checkpoints/FocalAug.pth"
        image_route = "VCA/PRÁCTICA 2/dataset/images"
        mask_route = "VCA/PRÁCTICA 2/dataset/masks"
        image_route = "None"
        mask_route = "None"
  
        print(f"Using default paths modelpath: {modelpath}, image_path: {image_route}")
    else:
        modelpath, image_route, mask_route = sys.argv[1], sys.argv[2], sys.argv[3]

    transform_basic = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((208, 312)),
        transforms.ToTensor()
    ])
    checkpoint = torch.load(modelpath, weights_only=False)

    if image_route == "None" or mask_route == "None":
        test_loader = checkpoint["test_set"]
    else:
        test_loader = DataLoader(OCTDataset(image_route, mask_route, transform=transform_basic), batch_size=16, shuffle=False)

    evaluate_model(checkpoint["full_model"], test_loader, checkpoint["history"], device, thresh=0.35)