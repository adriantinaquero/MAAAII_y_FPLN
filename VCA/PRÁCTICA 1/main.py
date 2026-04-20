import torch
from data_loader import load_dataset
from models import create_vgg
from train import train_model

# IMPLEMENTAR:
#   GUARDAR HISTORY, MÉTRICAS Y EJEMPLOS MAL CLASIFICADOS EN DISCO
#   ACCURACY, ESPECIFICIDAD Y SENSIBILIDAD POR CLASE Y GENERAL

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    route = "VCA/PRÁCTICA 1/dataset/ship.csv"
    route2 = "VCA/PRÁCTICA 1/dataset/docked.csv"

    batch_size = 128

    train_basic, train_aug, val_loader, test_loader = load_dataset(route, batch_size)
    docked_train_basic, docked_train_aug, docked_val_loader, docked_test_loader = load_dataset(route2, batch_size)


    # PREENTRENADA Y SIN AUGMENTATION
    print("PREENTRENADA Y SIN AUGMENTATION")
    model = create_vgg(device, pretrained=True)
    model, history = train_model(
        model,
        train_basic,
        val_loader,
        test_loader,
        device,
        epochs=50
    )

    torch.save({
        "model_state": model.state_dict(),
        "full_model": model,
        "history": history,
        "test_set": test_loader,
    }, "VCA/PRÁCTICA 1/checkpoints/PREENTRENADA_SIN_AUGMENTATION.pth")


    # PREENTRENADA Y CON AUGMENTATION
    print("PREENTRENADA Y CON AUGMENTATION")
    model = create_vgg(device, pretrained=True)
    model, history = train_model(
        model,
        train_aug,
        val_loader,
        test_loader,
        device,
        epochs=50
    )

    torch.save({
    "model_state": model.state_dict(),
    "full_model": model,
    "history": history,
    "test_set": test_loader,
    }, "VCA/PRÁCTICA 1/checkpoints/PREENTRENADA_CON_AUGMENTATION.pth")


    # NO PREENTRENADA Y SIN AUGMENTATION
    print("NO PREENTRENADA Y SIN AUGMENTATION")
    model = create_vgg(device, pretrained=False)
    model, history = train_model(
        model,
        train_basic,
        val_loader,
        test_loader,
        device,
        epochs=90
    )

    torch.save({
        "model_state": model.state_dict(),
        "full_model": model,
        "history": history,
        "test_set": test_loader,
    }, "VCA/PRÁCTICA 1/checkpoints/NO_PREENTRENADA_SIN_AUGMENTATION.pth")


    # NO PREENTRENADA Y CON AUGMENTATION
    print("NO PREENTRENADA Y CON AUGMENTATION")
    model = create_vgg(device, pretrained=False)
    model, history = train_model(
        model,
        train_aug,
        val_loader,
        test_loader,
        device,
        epochs=90
    )

    torch.save({
        "model_state": model.state_dict(),
        "full_model": model,
        "history": history,
        "test_set": test_loader,
    }, "VCA/PRÁCTICA 1/checkpoints/NO_PREENTRENADA_CON_AUGMENTATION.pth")

    # DOCKED PREENTRENADA Y SIN AUGMENTATION
    print("DOCKED PREENTRENADA Y SIN AUGMENTATION")
    model = create_vgg(device, pretrained=True)
    model, history = train_model(
        model,
        docked_train_basic,
        docked_val_loader,
        docked_test_loader,
        device,
        epochs=100
    )

    torch.save({
        "model_state": model.state_dict(),
        "full_model": model,
        "history": history,
        "test_set": docked_test_loader,
    }, "VCA/PRÁCTICA 1/checkpoints/DOCKED_PREENTRENADA_SIN_AUGMENTATION.pth")


    # DOCKED PREENTRENADA Y CON AUGMENTATION
    print("DOCKED PREENTRENADA Y CON AUGMENTATION")
    model = create_vgg(device, pretrained=True)
    model, history = train_model(
        model,
        docked_train_aug,
        docked_val_loader,
        docked_test_loader,
        device,
        epochs=100
    )

    torch.save({
    "model_state": model.state_dict(),
    "full_model": model,
    "history": history,
    "test_set": docked_test_loader,
    }, "VCA/PRÁCTICA 1/checkpoints/DOCKED_PREENTRENADA_CON_AUGMENTATION.pth")

    # DOCKED NO PREENTRENADA Y SIN AUGMENTATION
    checkpoint = torch.load("VCA/PRÁCTICA 1/checkpoints/NO_PREENTRENADA_SIN_AUGMENTATION.pth", weights_only=False)
    print("DOCKED NO PREENTRENADA Y SIN AUGMENTATION")
    model = checkpoint["full_model"]
    for param in model[:-4].parameters(): param.requires_grad = False
    model, history = train_model(
        model,
        docked_train_basic,
        docked_val_loader,
        docked_test_loader,
        device,
        epochs=50
    )
    torch.save({
    "model_state": model.state_dict(),
    "full_model": model,
    "history": history,
    "test_set": docked_test_loader,
    }, "VCA/PRÁCTICA 1/checkpoints/DOCKED_NO_PREENTRENADA_SIN_AUGMENTATION.pth")

    # DOCKED NO PREENTRENADA Y CON AUGMENTATION
    checkpoint = torch.load("VCA/PRÁCTICA 1/checkpoints/NO_PREENTRENADA_CON_AUGMENTATION.pth", weights_only=False)
    print("DOCKED NO PREENTRENADA Y CON AUGMENTATION")
    model = checkpoint["full_model"]
    for param in model[:-4].parameters(): param.requires_grad = False
    model, history = train_model(
        model,
        docked_train_aug,
        docked_val_loader,
        docked_test_loader,
        device,
        epochs=50
    )
    torch.save({
    "model_state": model.state_dict(),
    "full_model": model,
    "history": history,
    "test_set": docked_test_loader,
    }, "VCA/PRÁCTICA 1/checkpoints/DOCKED_NO_PREENTRENADA_CON_AUGMENTATION.pth")