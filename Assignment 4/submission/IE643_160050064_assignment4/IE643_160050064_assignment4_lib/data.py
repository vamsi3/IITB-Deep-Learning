## AUTHOR: Vamsi Krishna Reddy Satti


import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class NaturalImagesDataset(Dataset):
    """
    Dataset source: https://www.kaggle.com/prasunroy/natural-images
    """

    def __init__(self, root, category, train=True, transform=None):
        super().__init__()
        self.root = os.path.join(root, category)
        self.transform = transform
        self.filenames = sorted(os.listdir(self.root))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.filenames[idx])
        img = Image.open(filepath).convert('RGB')
        img = self.transform(img)
        return img
