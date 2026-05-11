import torch
from data_loader import load_dataset
from models import BaseLine
from train import train_model

# IMPLEMENTAR:
#   GUARDAR HISTORY, MÉTRICAS Y EJEMPLOS MAL CLASIFICADOS EN DISCO
#   ACCURACY, ESPECIFICIDAD Y SENSIBILIDAD POR CLASE Y GENERAL

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_route = "VCA/PRÁCTICA 2/dataset/images"
    mask_route = "VCA/PRÁCTICA 2/dataset/masks"

    batch_size = 32

    train_basic, train_aug, val_loader, test_loader = load_dataset(image_route, mask_route, batch_size)


    # PREENTRENADA Y SIN AUGMENTATION
    print("BASELINE")
    model = BaseLine(1, 1).to(device)
    model, history = train_model(
        model,
        train_basic,
        val_loader,
        test_loader,
        device,
        epochs=5
    )

    # torch.save({
    #     "model_state": model.state_dict(),
    #     "full_model": model,
    #     "history": history,
    #     "test_set": test_loader,
    # }, "VCA/PRÁCTICA 1/checkpoints/PREENTRENADA_SIN_AUGMENTATION.pth")