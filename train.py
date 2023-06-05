import os
import time
import ssl

from Trainer import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    runTrain()


def runTrain():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    timestampLaunch = timestampDate + '-' + timestampTime

    pathDirData = 'data'

    pathFileTrain = pathDirData+'\label_train_aug.txt'
    pathFileVal = pathDirData+'\label_test_aug.txt'

    nnClassCount = 2

    trBatchSize = 16
    trMaxEpoch = 15

    imgtransCrop = 300

    pathModel = None

    Trainer.train(pathDirData, pathFileTrain, pathFileVal, nnClassCount, trBatchSize, trMaxEpoch, imgtransCrop,
                  timestampLaunch, pathModel)


if __name__ == '__main__':
    main()
