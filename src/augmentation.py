from torchvision import transforms, datasets

MEAN = [0.485, 0.456, 0.406]
STDEV = [0.229, 0.224, 0.225]

class CustomRandomCrop:
    def __init__(self, crop_percent=0.5):
        self.crop_percent = crop_percent

    def __call__(self, img):
        width, height = img.size

        crop_width = int(width * self.crop_percent)
        crop_height = int(height * self.crop_percent)

        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)

        right = left + crop_width
        bottom = top + crop_height

        return img.crop((left, top, right, bottom))

train_transforms = transforms.Compose([
        transforms.RandomApply([CustomRandomCrop(crop_percent=0.6)], p=0.5),
        # transforms.RandomRotation(degrees = (-30, 30)),
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
