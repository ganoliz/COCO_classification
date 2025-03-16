import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CocoFourClassDataset(Dataset):
    def __init__(self, image_dir, annotations, transform=None):

        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, label = self.annotations[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
