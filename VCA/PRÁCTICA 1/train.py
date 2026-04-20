import torch
from torch import nn, optim
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, test_loader, device, epochs=5):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": []
    }
    
    val_loss = 0
    val_acc = 0

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        train_acc = 0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images).squeeze(1)
            preds = (outputs > 0.5).float()

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            size = labels.size(0)
            train_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += size

        train_loss = train_loss / len(train_loader)
        train_acc = correct / total

        print(f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()

                outputs = model(images).squeeze(1)
                preds = (outputs > 0.5).float()

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)


    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images).squeeze(1)
            preds = (outputs > 0.5).float()

            loss = criterion(outputs, labels)

            test_loss += loss.item()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_loader)
    test_acc = correct / total

    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    return model, history