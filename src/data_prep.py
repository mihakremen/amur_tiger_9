import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from augmentation import train_transforms, valid_transforms
from cfg import config


TRAIN_FOLDER = config["data"]["train_path"]
VAL_FOLDER = config["data"]["val_path"]
BATCH_SIZE= config["data"]["batch_size"]

train_dataset = datasets.ImageFolder(root = TRAIN_FOLDER, 
                                     transform=train_transforms)
valid_dataset = datasets.ImageFolder(root = VAL_FOLDER, 
                                     transform=valid_transforms)

num_gpus = torch.cuda.device_count()
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2*num_gpus, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2*num_gpus, pin_memory=True)