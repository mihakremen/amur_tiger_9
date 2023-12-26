# Импорт библиотек
import torch
import argparse
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from resnet18 import Model
from augmentation import valid_transforms

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Path to the image file", default = None)
parses.add_argument("--model", help="Path to the model file", default = None)
args = parser.parse_args()


# Подгрузка модели
if args.model is None:
    model_path = "https://www.dropbox.com/scl/fi/h4o6g18lgpfigm3sfqqxu/model_best_f1.pth.tar?rlkey=zu4d6gi9e50od2bywewj8nn1u&dl=1"
else:
    model_path = args.model
    
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

net = Model()
net.load_state_dict(checkpoint["state_dict"])
net.eval()

# Загрузка изображения
if args.image is None:
    image_url = 'https://rgo.ru/upload/s34web.imageadapter/3391b95c081666cc16ceb5b195d7ccf4/oleg_bogdanov_amurskiy_tigr_592345.jpg'
    urllib.request.urlretrieve(image_url,"image.png")
    image = Image.open("image.png")
else:
    image = Image.open(args.image)

# Инференс изображения через модель
image = valid_transforms(image).unsqueeze(0)

# Получение сырого предскзаания
with torch.no_grad():
    result, conf = net.predict(image)

# Обработка сырого предсказания до до вида 'left side'
classes = ['front', 'left side', 'other', 'right side']
print(classes[result], conf)
