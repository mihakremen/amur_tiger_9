from transformers import ViTImageProcessor
import torch
from data_creation import dataset_train, dataset_val


model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTImageProcessor.from_pretrained(
model_id
)

def preprocess(batch):
    # take a list of PIL images and turn them to pixel values
    inputs = feature_extractor(
        batch['image'],
        return_tensors='pt'
    )
    # include the labels
    inputs['label'] = batch['label']
    return inputs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prepared_train = dataset_train.with_transform(preprocess)
prepared_val = dataset_val.with_transform(preprocess)
# prepared_test = dataset_test.with_transform(preprocess)

