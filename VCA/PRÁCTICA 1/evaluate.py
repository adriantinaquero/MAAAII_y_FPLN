import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(model, test_loader, history, device, num_classes=2):

    # EVALUACIÓN CUANTITATIVA

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)

    # mostramos la matriz de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # Accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"Accuracy global: {accuracy:.4f}")

    # Sensibilidad y especificidad por clase
    print("\nSensibilidad y Especificidad por clase:")

    for i in range(num_classes):

        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        print(f"Clase {i}:             "
              f"Sensibilidad={sensitivity:.4f}         "
              f"Especificidad={specificity:.4f}")
        
    epochs = range(1, len(history["train_loss"]) + 1)

    # curva de Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], 'k', label="Train")
    plt.plot(epochs, history["test_loss"], 'r', label="Test")
    plt.title("Loss")
    plt.legend()
    plt.show()

    # curva de Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], 'k', label="Train")
    plt.plot(epochs, history["test_acc"], 'r', label="Test")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

   # EVALUACIÓN CUALITATIVA

    # mostramos algunos ejemplos de cada clase predecidos incorrectamente
    for clase in range(num_classes):
        print(f"\nClase {clase} - Errores:")
        show_examples(model, test_loader, clase)



def show_examples(model, loader, class_id, device, num_images=5):

    model.eval()
    shown = 0

    plt.figure(figsize=(10,2))

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(len(images)):

                if labels[i] == class_id and preds[i] != labels[i]:
                    plt.subplot(1, num_images, shown+1)
                    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
                    plt.title(f"Predicción:{preds[i].item()}")
                    plt.axis("off")

                    shown += 1

                    if shown == num_images:
                        plt.show()
                        return