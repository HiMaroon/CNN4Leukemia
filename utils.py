from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
from typing import Sequence
import torchvision.transforms as transforms


class DatasetGenerator(Dataset):

    def __init__(self, pathDatasetFile, transCrop):

        self.listImagePaths = []
        self.listImageLabels = []
        self.transcrop=transCrop

        fileDescriptor = open(pathDatasetFile, "r", encoding='utf-8')

        line = True

        while line:

            line = fileDescriptor.readline()

            if line:
                lineItems = line.split()

                imagePath = lineItems[0]

                imageLabel = int(lineItems[1])

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)

        fileDescriptor.close()

    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]

        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.tensor(self.listImageLabels[index])

        transformList = [transforms.Resize([self.transcrop, self.transcrop]),
                             transforms.ToTensor()]

        transformSequence = transforms.Compose(transformList)
        imageData = transformSequence(imageData)

        return imageData, imageLabel

    def __len__(self):

        return len(self.listImagePaths)


class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)