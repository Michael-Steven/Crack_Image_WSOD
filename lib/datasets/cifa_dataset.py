import torchvision
import torch
from torchvision import transforms

CIFAR_PATH = "/home/syb/documents/Crack_Image_WSOD/data/"
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
num_workers = 2


def cifar10_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    cifar10_training = torchvision.datasets.CIFAR10(
        root=CIFAR_PATH, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        cifar10_training, batch_size=16, shuffle=True, num_workers=num_workers)

    cifar10_testing = torchvision.datasets.CIFAR10(
        root=CIFAR_PATH, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        cifar10_testing, batch_size=1, shuffle=False, num_workers=num_workers)

    return trainloader, testloader
