from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(batch_size=128, img_size=28, data_root="./data"):
    """加载MNIST训练数据集"""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]，匹配高斯噪声分布
    ])

    dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    return dataloader, dataset
