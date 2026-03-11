import torch
from torch import nn, optim
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, test_loader, device, epochs=5):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": []
    }
    

    for epoch in range(epochs):

        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = torch.tensor(labels)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(test_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


    # model.eval()
    # test_loss = 0
    # correct = 0
    # total = 0

    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)

    #         outputs = model(images)
    #         loss = criterion(outputs, labels)

    #         test_loss += loss.item()

    #         _, preds = torch.max(outputs, 1)
    #         correct += (preds == labels).sum().item()
    #         total += labels.size(0)

    # test_loss /= len(test_loader)
    # test_acc = correct / total

    # history["test_loss"].append(test_loss)
    # history["test_acc"].append(test_acc)

    return model, history