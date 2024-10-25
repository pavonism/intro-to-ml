import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model_training.audio_dataset import AudioDataset
from model_training.neural_network import Net
from model_training.train_loop import Loop
from model_training.test_methods import Test


def run_training(
    image_train_path: str,
    image_val_path: str,
    image_test_path: str,
    batch_size: int = 32,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(
        "Running training on GPU..."
        if torch.cuda.is_available()
        else "Running training on CPU..."
    )

    transform = transforms.Compose(
        [
            transforms.Resize((300, 400)),
            transforms.ToTensor(),
        ]
    )

    valset = AudioDataset(image_val_path, transform=transform)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    trainset = AudioDataset(image_train_path, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    model = Net()
    model.to(device)

    PATH = "../models/first-milestone.pth"

    model = Loop(
        model=model,
        path=PATH,
        train_loader=trainloader,
        validation_loader=valloader,
        device=device,
        num_epochs=5,
    )

    Test(image_test_path, transform, device, model)
