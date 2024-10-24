import torch
from PIL import Image
import glob


def wich_class(image_path):
    if("Class1" in image_path):
        return "Class1"
    elif("Class0" in image_path):
        return "Class0"
    else:
        return "Error"

def Test(image_test_path, transform , device , model):
    images = glob.glob(f"{image_test_path}/**/**.png", recursive=True)

    for image in images:
        image_tensor = preprocess_image(image, transform)
        probabilities = predict(model, image_tensor, device)

        print(wich_class(image))
        print(probabilities)

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()
