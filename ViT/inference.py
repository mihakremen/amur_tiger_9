# Установка библиотек
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import urllib.request
from PIL import Image

# Подгрузка модели и feature extractor
model = ViTForImageClassification.from_pretrained("Goolissimo/Tigers_side_ViT")
feature_extractor = ViTImageProcessor.from_pretrained('Goolissimo/Tigers_side_ViT')
model.eval()

# Загрузка изображения
image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS1ylvBwKTsuYIwGa7WPW4lA7xV2JnsEuda4aNOP5ARIA&s'
urllib.request.urlretrieve(image_url, "image.jpg")
img = Image.open("image.jpg")

# Инференс изображения через модель
inputs = feature_extractor(img, return_tensors="pt")

# Получение сырого предскзаания
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

# Обработка сырого предсказания до вида 'left'
predicted_label = outputs.logits.argmax(-1).item()
labels = ['front', 'left', 'other', 'right']
print(labels[predicted_label])
