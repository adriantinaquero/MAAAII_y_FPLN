import sys
import torch
from dataset import ShipData
from evaluate import evaluate_model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if len(sys.argv) != 3:
        print("Usage: python execute_test.py modelpath datasetpath")
        modelpath = "VCA/PRÁCTICA 1/checkpoints/PREENTRENADA_SIN_AUGMENTATION.pth"
        datasetpath = "VCA/PRÁCTICA 1/P1-test-images/ship.csv" 
        print(f"Using default paths modelpath: {modelpath}, datasetpath: {datasetpath}")
    else:
        modelpath, datasetpath = sys.argv[1], sys.argv[2]

    transform_basic = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,
          std=0.225)
    ])
    checkpoint = torch.load(modelpath, weights_only=False)

    if datasetpath == "None":
        test_loader = checkpoint["test_set"]
    else:
        test_loader = DataLoader(ShipData(datasetpath, transform_basic), batch_size=128, shuffle=False)

    evaluate_model(checkpoint["full_model"], test_loader, checkpoint["history"], device)