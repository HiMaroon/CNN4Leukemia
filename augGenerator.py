import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
import os
import random

from utils import MyRotateTransform


pathDirData = 'data'
pathImg = 'data\img'

transCrop=300
transformList = [transforms.Resize([transCrop, transCrop]),
                 transforms.ColorJitter(brightness=(0.9,1.1)),
                 transforms.ColorJitter(contrast=(0.9,1.1)),
                 transforms.ColorJitter(saturation=(0.9,1.1)),
                 transforms.RandomVerticalFlip(),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomAdjustSharpness(sharpness_factor=2),
                 MyRotateTransform([0, 90, 180, 270]),
                 ]

transformSequence = transforms.Compose(transformList)

def DataPicGenerator(pathImg):
    listdir = os.listdir(pathImg)
    for filename in listdir:
        imagePath=pathImg+'\\'+filename
        imageData = Image.open(imagePath).convert('RGB')

        number = random.random()
        if number >= 0.3:
            for i in range(10):
                convertedImg=transformSequence(imageData)
                convertedImg=np.array(convertedImg)
                convertedImg = cv2.cvtColor(convertedImg, cv2.COLOR_BGR2RGB)
                cv2.imwrite(pathDirData+'\Training\\'+filename[:-4]+'_'+str(i)+'.png',convertedImg)
        else:
            for i in range(10):
                convertedImg = transformSequence(imageData)
                convertedImg = np.array(convertedImg)
                convertedImg = cv2.cvtColor(convertedImg, cv2.COLOR_BGR2RGB)
                cv2.imwrite(pathDirData + '\Validation\\' + filename[:-4] + '_' + str(i) + '.png', convertedImg)

def DataTxtGenarator(outputdir):
    imgListTrain = os.listdir(outputdir+'/Training')
    imgListTrain = [item.replace('.png', '') for item in imgListTrain]
    with open(outputdir + "/label_train_aug.txt", "w") as f:
        for name in imgListTrain:
            label=name[-3]
            f.write(outputdir+'/Training/'+name+'.png '+label+'\n')

    imgListVal = os.listdir(outputdir + '/Validation')
    imgListVal = [item.replace('.png', '') for item in imgListVal]
    with open(outputdir + "/label_test_aug.txt", "w") as f:
        for name in imgListVal:
            label = name[-3]
            f.write(outputdir + '/Validation/' + name + '.png ' + label + '\n')



DataPicGenerator(pathImg)
DataTxtGenarator(pathDirData)
