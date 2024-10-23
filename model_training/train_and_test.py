import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import glob

from model_training.audio_dataset import AudioDataset
from model_training.neural_network import NeuralNetwork
from model_training.spects_loader import get_class_for_file
from model_training.test_methods import predict, preprocess_image
from model_training.train_loop import train_loop

# TODO: Refactor this code

batch_size = 4

# audio_path = './daps/train'
image_path = "./Simple_Model/Scripts/Spectograms"

# test_audio_path = './daps/test'
image_test_path = "./Simple_Model/Scripts/Spectograms_test"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

# SpectsLoader.SpectsLoader(audio_path , image_path)
trainset = AudioDataset(image_path, transform=transform)

# SpectsLoader.SpectsLoader(test_audio_path , image_test_path)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

model = NeuralNetwork()

PATH = "./Simple_Model/cifar_net.pth"

model = train_loop(
    model=model,
    PATH=PATH,
    trainloader=trainloader,
    device=device,
    num_epochs=0,
    load=True,
)

images = glob.glob(f"{image_test_path}/**/**.png", recursive=True)

for image in images:
    original_image, image_tensor = preprocess_image(image, transform)
    probabilities = predict(model, image_tensor, device)

    print(get_class_for_file((image.split("\\"))[2]))
    print(probabilities)
