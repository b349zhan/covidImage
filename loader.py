import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from config import *
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
class CovidDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).type(torch.DoubleTensor)
        label = torch.tensor(self.img_labels.iloc[idx, 1])
        if self.transform:
           image = self.transform(image)

        return image, label
def getLoader():

    givenTrainData = CovidDataset(annotations_file=TRAINING_Y_PATH, img_dir=TRAINING_X_PATH)
    trainData, valData = random_split(givenTrainData, [TRAIN_SIZE, VAL_SIZE],generator=torch.Generator().manual_seed(42) )

    BATCH_SIZE = 32
    train_dataloader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(valData, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, val_dataloader