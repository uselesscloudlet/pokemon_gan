import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def create_dl(data_path: str, img_size: int, bs: int, workers: int) -> DataLoader:
    nn_transforms = transforms.Compose([
        transforms.RandomRotation(20, fill=255),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomPosterize(bits=2),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomAutocontrast(),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = dset.ImageFolder(root=data_path,
                               transform=nn_transforms)

    dataloader = DataLoader(dataset,
                            batch_size=bs,
                            shuffle=True,
                            num_workers=workers)

    return dataloader
