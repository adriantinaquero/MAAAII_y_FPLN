import torch
from torch import nn, optim

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.15, gamma=1.5, reduction='mean', pos_weight = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, outputs, labels):
        bce = self.crit(outputs, labels)
        p_t = torch.exp(-bce)                          
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def train_model(model, train_loader, val_loader, test_loader, device, epochs=5):

    pos_weight = compute_pos_weight(train_loader).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = FocalLoss(0.25, 3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_dice": [],
        "val_dice": [],
        "test_dice": []
    }
    
    val_loss = 0
    val_dice = 0

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        train_dice = 0
        size = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).float()
            batchsize = labels.size(0)
            optimizer.zero_grad()
            outputs = model(images)
            preds = get_segmentation_masks(outputs, 0.5)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batchsize
            train_dice += dice_score(outputs, labels).item() * batchsize
            size += batchsize
        train_loss /= size
        train_dice /= size


        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()

                outputs = model(images)
                preds = get_segmentation_masks(outputs, 0.5)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_dice += dice_score(outputs, labels).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Train Dice: {train_dice:.4f} | Val dice: {val_dice:.4f}")

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)


    model.eval()
    test_loss = 0
    test_dice = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            preds = get_segmentation_masks(outputs, 0.5)

            loss = criterion(outputs, labels)

            test_loss += loss.item()
            test_dice += dice_score(preds, labels)


    test_loss /= len(test_loader)
    test_dice /= len(test_loader)

    history["test_loss"].append(test_loss)
    history["test_dice"].append(test_dice)

    return model, history

def get_segmentation_masks(outputs, threshold=0.5):
    probs = torch.sigmoid(outputs)
    masks = (probs > threshold)*1.0
    return masks

def compute_pos_weight(loader):
    pos = 0
    neg = 0
    for _, masks in loader:
        pos += (masks == 1).sum().item()
        neg += (masks == 0).sum().item()
    print(f"Positive pixels: {pos}, Negative pixels: {neg}, Ratio: {neg/pos:.1f}")
    return torch.tensor([neg / pos])

def dice_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    return (2 * intersection + 1) / (preds.sum() + targets.sum() + 1)