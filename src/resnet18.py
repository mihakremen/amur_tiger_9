import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class Model(nn.Module):
    def __init__(self, num_classes = 4):
        super().__init__()
        self.resnet = resnet18(weights = ResNet18_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
    
    def predict(self, x):
        x = self.resnet(x)
        x = nn.functional.softmax(x, dim = 1)
        logit, idx = torch.max(x, dim = 1)
        return idx, logit