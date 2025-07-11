import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
# Note - you must have torchvision installed for this example
from torchvision.datasets import CelebA

from utils.utils import rescaling
class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CelebA64DataModule(L.LightningDataModule):
    def __init__(self, batch_size, size, num_workers=0, data_dir: str = "./"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        self.fid_transform = transforms.Compose([
            CropCelebA64(),
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        # download
        CelebA(self.data_dir, split='train', download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.train = CelebA(self.data_dir, split='train', transform=self.transform)
        self.fid = CelebA(self.data_dir, split='train', transform=self.fid_transform)

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def fid_dataloader(self):
        return DataLoader(self.fid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)