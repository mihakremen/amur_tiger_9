from torchvision import transforms, datasets

MEAN = [0.485, 0.456, 0.406]
STDEV = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
        transforms.RandomRotation(degrees = (-30, 30)),
#         transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
#         transforms.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.3), scale=(0.6, 0.8)),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STDEV)
        ])

valid_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STDEV)
    ])