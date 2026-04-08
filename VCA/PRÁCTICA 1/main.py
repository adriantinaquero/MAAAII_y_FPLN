import torch
from data_loader import load_dataset
from models import create_vgg
from train import train_model
from evaluate import evaluate_model
import numpy as np

# IMPLEMENTAR:
#   GUARDAR HISTORY, MÉTRICAS Y EJEMPLOS MAL CLASIFICADOS EN DISCO
#   ACCURACY, ESPECIFICIDAD Y SENSIBILIDAD POR CLASE Y GENERAL

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    route = "VCA/PRÁCTICA 1/dataset/ship.csv"

    batch_size = 256

    train_basic, train_aug, val_loader, test_loader = load_dataset(route, batch_size)

    
    # PREENTRENADA Y SIN AUGMENTATION
    model = create_vgg(device, pretrained=True)
    print("PREENTRENADA Y SIN AUGMENTATION")
    model, history = train_model(
        model,
        train_basic,
        val_loader,
        test_loader,
        device,
        epochs=15
    )
    evaluate_model(model, test_loader, history, device)


    # # PREENTRENADA Y CON AUGMENTATION
    # print("PREENTRENADA Y CON AUGMENTATION")
    # model = create_vgg(device, pretrained=True)
    # model, history = train_model(
    #     model,
    #     train_aug,
    #     val_loader,
    #     test_loader,
    #     device,
    #     epochs=5
    # )
    # evaluate_model(model, test_loader, history, device)


    # # NO PREENTRENADA Y SIN AUGMENTATION
    # print("NO PREENTRENADA Y SIN AUGMENTATION")
    # model = create_vgg(device, pretrained=False)
    # model, history = train_model(
    #     model,
    #     train_basic,
    #     val_loader,
    #     test_loader,
    #     device,
    #     epochs=5
    # )
    # evaluate_model(model, test_loader, history, device)


    # # NO PREENTRENADA Y CON AUGMENTATION
    # print("NO PREENTRENADA Y CON AUGMENTATION")
    # model = create_vgg(device, pretrained=False)
    # model, history = train_model(
    #     model,
    #     train_aug,
    #     val_loader,
    #     test_loader,
    #     device,
    #     epochs=5
    # )
    # evaluate_model(model, test_loader, history, device)