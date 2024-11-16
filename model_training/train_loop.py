from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm


def save_results(path: str, train_loss: float, val_loss: float):
    with open(f"{path}/train_losses.csv", "a") as f:
        f.write("%s\n" % train_loss)

    with open(f"{path}/val_losses.csv", "a") as f:
        f.write("%s\n" % val_loss)


def Loop(
    model: nn.Module,
    path: str,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        min_lr=1e-6,
        verbose=True,
    )

    train_losses, val_losses = [], []
    val_loss_rising = 0

    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(validation_loader.dataset)

        if epoch > 0:
            val_loss_rising = 0 if val_loss < val_losses[-1] else val_loss_rising + 1

        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch} Done after {time.time() - start_time} seconds")
        save_results(path, train_loss, val_loss)

        if val_loss_rising > 5:
            print("Validation score increased five times in a row. Early stopping...")
            break

    print("Finished Training")
    torch.save(model.state_dict(), f"{path}/model.pth")

    return model
