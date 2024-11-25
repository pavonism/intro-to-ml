from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class AudioDataset(Dataset):
    def __init__(self, data_dir, transform=None, device=None):
        self.data = ImageFolder(data_dir, transform=transform)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.device:
            image = image.to(self.device, non_blocking=True)

        return image, label

    @property
    def classes(self):
        return self.data.classes
