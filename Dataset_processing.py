from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
from torchvision.transforms import functional as TF
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class MyDataset(Dataset):
    def __init__(self, root, transform=None, batch_size=4, shuffle=True):
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.dataset = datasets.ImageFolder(root=root, transform=self.transform)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

class CombinedDataset(Dataset):
    def __init__(self, vision_dataset, touch_dataset, transform=None, train=True):
        self.vision_dataset = vision_dataset
        self.touch_dataset = touch_dataset
        self.transform = transform
        self.train = train
        self.labels = torch.tensor([label for _, label in self.vision_dataset])
        self.data = [(vision_data, touch_data) for vision_data, touch_data in zip(self.vision_dataset, self.touch_dataset)]

    def __len__(self):
        return len(self.vision_dataset)

    def __getitem__(self, idx):
        vision_item = self.vision_dataset[idx]
        touch_item = self.touch_dataset[idx]
        touch_item_resized = F.interpolate(touch_item[0].unsqueeze(0), size=(256, 256), mode='bilinear',
                                           align_corners=False).squeeze(0)
        touch_item_resized = F.interpolate(touch_item_resized.unsqueeze(0), size=(224, 224), mode='bilinear',
                                           align_corners=False).squeeze(0)
        vision_item_resized = F.interpolate(vision_item[0].unsqueeze(0), size=(224, 224), mode='bilinear',
                                           align_corners=False).squeeze(0)

        return vision_item_resized, touch_item_resized, vision_item[1]