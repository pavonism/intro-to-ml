import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import glob

import Dataset
import Model
import Trainloop
import SpectsLoader
import Test_funs

batch_size = 4


#audio_path = './daps/train'
image_path = './Simple_Model/Scripts/Spectograms'

#test_audio_path = './daps/test'
image_test_path = './Simple_Model/Scripts/Spectograms_test'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

#SpectsLoader.SpectsLoader(audio_path , image_path)
trainset = Dataset.AudioDataset(image_path , transform = transform)

#SpectsLoader.SpectsLoader(test_audio_path , image_test_path)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

model = Model.Net()

PATH = './Simple_Model/cifar_net.pth'

model = Trainloop.Loop(model = model, PATH = PATH , trainloader = trainloader , 
                       device = device , num_epochs = 0 , load = True)

images = glob.glob(f"{image_test_path}/**/**.png", recursive=True)

for image in images:
    original_image, image_tensor = Test_funs.preprocess_image(image, transform)
    probabilities = Test_funs.predict(model, image_tensor, device)

    print(SpectsLoader.If_Class1((image.split('\\'))[2]))
    print(probabilities)