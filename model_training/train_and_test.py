import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import audio_dataset
import neural_network
import train_loop

batch_size = 4


image_train_path = './model_training/Spectograms_train'
image_test_path = './model_training/Spectograms_test'
image_val_path = './model_training/Spectograms_val'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available()) # If prints TRUE, you should be using gpu

transform = transforms.Compose([
    transforms.Resize((300, 400)),
    transforms.ToTensor(),
])

valset = audio_dataset.AudioDataset(image_val_path , transform = transform)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

trainset = audio_dataset.AudioDataset(image_train_path , transform = transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

model = neural_network.Net()
model.to(device)

PATH = './Simple_Model/cifar_net.pth'

model = train_loop.Loop(model = model, PATH = PATH , trainloader = trainloader , valloader = valloader , 
                       device = device , num_epochs = 5 , load = True) # If you want to load weights from previous training set load = True, it will load file previously saved in work folder, or not if there isnt one



#Test_funs.Test(image_test_path, transform , device , model)
