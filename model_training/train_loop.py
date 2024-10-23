import torch
import torch.nn as nn
import torch.optim as optim
import os


def train_loop(model, PATH, trainloader, device, num_epochs, load):
    if os.path.exists(PATH) == True and load:
        model.load_state_dict(torch.load(PATH, weights_only=True))
        model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses = []

    for _ in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(trainloader.dataset)
        train_losses.append(train_loss)

    print("Finished Training")

    for losses in train_losses:
        print(losses)

    torch.save(model.state_dict(), PATH)

    return model
