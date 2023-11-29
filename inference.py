# Импорт библиотек
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Подгрузка модели
model_path = "https://www.dropbox.com/scl/fi/h4o6g18lgpfigm3sfqqxu/model_best_f1.pth.tar?rlkey=zu4d6gi9e50od2bywewj8nn1u&dl=1"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
class Model(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def predict(self, x):
        x = self.resnet(x)
        x = nn.functional.softmax(x, dim=1)
        logit, idx = torch.max(x, dim=1)
        return idx, logit
net = Model()
net.load_state_dict(checkpoint["state_dict"])
net.eval()

# Загрузка изображения
image_url = 'https://rgo.ru/upload/s34web.imageadapter/3391b95c081666cc16ceb5b195d7ccf4/oleg_bogdanov_amurskiy_tigr_592345.jpg'
urllib.request.urlretrieve(image_url,"image.png")
image = Image.open("image.png")
plt.imshow(image)

# Инференс изображения через модель
mean=[0.4277, 0.3982, 0.3743]
stdev=[0.2377, 0.2300, 0.2229]
valid_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, stdev)
    ])
image = valid_transforms(image).unsqueeze(0)

# Получение сырого предскзаания
with torch.no_grad():
    result, conf = net.predict(image)
print(result, conf)

# Обработка сырого предсказания до до вида 'left side'
classes = ['front', 'left side', 'other', 'right side']
print(classes[result])
