import torch
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST

from utils.utils import rescaling, ResumableDataLoader

class Binarize(object):
    def __call__(self, pic):
        new_pic = torch.round(pic)
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, data_dir: str = "./", binary=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        if binary:
            self.transform = transforms.Compose([transforms.ToTensor(), Binarize()])
            self.fid_transform = transforms.Compose([transforms.ToTensor(), Binarize()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), rescaling])
            self.fid_transform = transforms.ToTensor()

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train = MNIST(self.data_dir, train=True, transform=self.transform)
            self.fid = MNIST(self.data_dir, train=True, transform=self.fid_transform)
            self.test = MNIST(self.data_dir, train=False, transform=self.transform)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self, shuffle=True):
        return ResumableDataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def fid_dataloader(self):
        return DataLoader(self.fid, batch_size=500, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
