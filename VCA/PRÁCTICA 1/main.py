import torch
from data_loader import load_dataset
from models import create_vgg
from train import train_model
from evaluate import evaluate_model
import numpy as np

# IMPLEMENTAR:
#   DIVIDIR .CSV EN TRAIN, VAL Y TEST
#   ENTRENAR CON CONJUNTO DE VALIDACIÓN
#   ACCURACY, ESPECIFICIDAD Y SENSIBILIDAD POR CLASE Y GENERAL

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    route = "VCA/PRÁCTICA 1/dataset/ship.csv"

    train_basic, train_aug, val_loader, test_loader = load_dataset(route)


    model = create_vgg(device, pretrained=True)

    model, history = train_model(
        model,
        train_basic,
        val_loader,
        test_loader,
        device,
        epochs=5
    )

    evaluate_model(model, test_loader, device)