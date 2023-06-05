import os
from Trainer import Trainer


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    runTest()

def runTest():
    pathDirData = 'data'
    pathFileTest = pathDirData+'\label_test.txt'
    path_excel = 'results\\230413.xlsx'
    nnClassCount = 2
    trBatchSize = 100
    imgtransCrop = 300

    modelname='m-98.67788461538461-20230604-104302.pth'
    pathModel = './models/' + modelname + '.tar'

    Trainer.test(pathDirData, pathFileTest, pathModel, nnClassCount, trBatchSize, imgtransCrop, path_excel)


if __name__ == '__main__':
    main()
