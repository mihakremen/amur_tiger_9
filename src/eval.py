# Установка библиотек
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from data_preprocessing import device, feature_extractor, prepared_train, prepared_val #, prepared_test
from torch.utils.data import Dataset, DataLoader
from data_creation import dataset_val
from sklearn.metrics import f1_score
# Подгрузка модели и feature extractor
model = ViTForImageClassification.from_pretrained("Goolissimo/Tigers_side_ViT")
feature_extractor = ViTImageProcessor.from_pretrained('Goolissimo/Tigers_side_ViT')


model.eval()

predicted_labels=[]
y_true = []
with torch.no_grad():
    for img, label in zip(dataset_val['image'],dataset_val['label']):
        inputs = feature_extractor(img, return_tensors="pt")
        outputs = model(**inputs)
        predicted_labels.append(outputs.logits.argmax(-1).item())
        y_true.append(label)

print ('взвешенная f1 на данном датасете:', f1_score(y_true, predicted_labels, average='weighted'))

