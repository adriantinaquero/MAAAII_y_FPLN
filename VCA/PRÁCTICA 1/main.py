import torch
from data_loader import load_dataset
from models import load_dataset, create_vgg
from train import train_model
from evaluate import evaluate_model

# IMPLEMENTAR:
#   DIVIDIR .CSV EN TRAIN, VAL Y TEST
#   ENTRENAR CON CONJUNTO DE VALIDACIÓN
#   ACCURACY, ESPECIFICIDAD Y SENSIBILIDAD POR CLASE Y GENERAL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

route = "C:/Users/Usuario/Documents/adrispul/Universidad/INTELIGENCIA ARTIFICIAL/3º/2º SEMESTRE/VCA/PRÁCTICAS/PRÁCTICA 1/dataset"

data = load_dataset(route)

# model = create_vgg(pretrained=True)

# history = train_model(
#     model,
#     train_loader,
#     optimizer,
#     device,
#     epochs=5
# )

# evaluate_model(model, test_loader, device)