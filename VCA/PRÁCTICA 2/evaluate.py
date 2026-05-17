import torch
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from train import get_segmentation_masks

def evaluate_model(model, test_loader, history, device, num_classes=2, thresh=0.5):

    # EVALUACIÓN CUANTITATIVA

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            preds = get_segmentation_masks(outputs, thresh)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
        
    accuracy = np.mean(all_preds == all_labels)
    print(f"Accuracy global: {accuracy:.4f}")

    # print("\nSensibilidad y Especificidad por clase:")

    TP = np.diag(cm)
    FN = cm.sum(axis=1) - TP
    FP = cm.sum(axis=0) - TP
    TN = cm.sum() - (TP + FN + FP)

    sensitivity  = np.where(TP + FN > 0, TP / (TP + FN), 0.0)
    specificity  = np.where(TN + FP > 0, TN / (TN + FP), 0.0)
    precision    = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
    vpn          = np.where(TN + FN > 0, TN / (TN + FN), 0.0)
    iou          = np.where(TP + FP + FN > 0, TP / (TP + FP + FN), 0.0)
    dice         = np.where(2*TP + FP + FN > 0, (2*TP) / (2*TP + FP + FN), 0.0)
    balanced_acc = (sensitivity + specificity) / 2

    print(f"\n{'Metric':<25} {'Clase 0':>10} {'Clase 1':>10} {'Macro avg':>10}")
    print("-" * 58)
    print(f"{'TP':<25} {TP[0]:>10} {TP[1]:>10}")
    print(f"{'FP':<25} {FP[0]:>10} {FP[1]:>10}")
    print(f"{'FN':<25} {FN[0]:>10} {FN[1]:>10}")
    print(f"{'TN':<25} {TN[0]:>10} {TN[1]:>10}")
    print("-" * 58)
    print(f"{'Sensibilidad':<25} {sensitivity[0]:>10.4f} {sensitivity[1]:>10.4f} {sensitivity.mean():>10.4f}")
    print(f"{'Especificidad':<25} {specificity[0]:>10.4f} {specificity[1]:>10.4f} {specificity.mean():>10.4f}")
    print(f"{'Precisión':<25} {precision[0]:>10.4f} {precision[1]:>10.4f} {precision.mean():>10.4f}")
    print(f"{'VPN':<25} {vpn[0]:>10.4f} {vpn[1]:>10.4f} {vpn.mean():>10.4f}")
    print(f"{'IoU':<25} {iou[0]:>10.4f} {iou[1]:>10.4f} {iou.mean():>10.4f}")
    print(f"{'Dice':<25} {dice[0]:>10.4f} {dice[1]:>10.4f} {dice.mean():>10.4f}")
    print(f"{'Balanced Accuracy':<25} {balanced_acc[0]:>10.4f} {balanced_acc[1]:>10.4f}")

    print(f"{thresh} & {sensitivity.mean()} & {specificity.mean()}\
           & {precision.mean()} & {vpn.mean()} & {iou.mean()}, &\
              {dice.mean()} & {balanced_acc.mean()}")
    epochs = len(history["val_loss"])
    val_range = range(1, epochs + 1)
    train_range = np.linspace(1, epochs, len(history["train_loss"]))

    plt.figure()
    plt.plot(train_range, history["train_loss"], 'k', label="Train")
    plt.plot(val_range, history["val_loss"], 'r', label="Val")
    plt.title("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_range, history["train_loss"], 'k', label="Train")
    plt.plot(val_range, history["val_loss"], 'r', label="Val")
    plt.title("LogLoss")
    plt.yscale("log")
    plt.legend()
    plt.show()
    images = np.concatenate([[pair for pair in zip(*batch)] for batch in test_loader])
    samples = [images[1]] + rd.choices(images, k=3)
    model.eval()
    with torch.no_grad():
        for sample in samples:
            image, gt = sample
            image_input = torch.from_numpy(image).unsqueeze(0).float().to(device)

            output = model(image_input)
            pred = get_segmentation_masks(output, thresh)
            show_result(
                orig = image,
                gt = gt,
                prediction = pred.cpu().squeeze(1).numpy(),
            )

def show_result(orig, gt, prediction, title=None):
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    ax = axes.ravel()
    images = [orig, gt, prediction, orig*prediction]
    titles = ['Orig', 'Gt', 'Result', 'Overlap']
    for i, (im, tit) in enumerate(zip(images, titles)):
        ax[i].imshow(im.transpose(1, 2, 0), cmap='gray')
        ax[i].set_title(tit)
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()