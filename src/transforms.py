from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224):
    """
    Strong training augmentation requested for the final pipeline.

    This matches the transform block with:
    - vertical flip p=0.5
    - ColorJitter 0.2 / 0.2 / 0.2 / 0.05
    - RandomGrayscale
    - GaussianBlur
    - RandomErasing
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.65, 1.0),
            ratio=(0.8, 1.25),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.5),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
        transforms.RandomErasing(
            p=0.1,
            scale=(0.02, 0.1),
        ),
    ])


def get_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])
