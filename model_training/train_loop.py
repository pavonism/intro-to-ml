import torch
import torch.nn as nn
import torch.optim as optim
import os
import time


def Loop(model , PATH , trainloader , valloader , device , num_epochs , load):

    if(os.path.exists(PATH) == True and load):
        model.load_state_dict(torch.load(PATH, weights_only=True))
        model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses , val_losses = [] , []

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch} Started")
        
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

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in valloader:
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
            
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(valloader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} Done after {time.time() - start_time} seconds")

    print('Finished Training')

    #for val_loss , train_loss in val_losses , train_losses:
    #    print(val_loss , "\n" , train_loss , "\n")

    torch.save(model.state_dict(), PATH)

    return model