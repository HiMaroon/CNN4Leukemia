import os
import numpy as np
import time
import openpyxl as op
import xlrd

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn import metrics

import matplotlib.pyplot as plt

from model import get_model

from utils import  DatasetGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Trainer():

    def train(pathDirData, pathFileTrain, pathFileVal, nnClassCount, trBatchSize,
              trMaxEpoch, transCrop, launchTimestamp, pathModel):

        model = get_model()

        model = torch.nn.DataParallel(model).cuda()

        if pathModel:
            modelCheckpoint = torch.load(pathModel)
            model.load_state_dict(modelCheckpoint['state_dict'])

        datasetTrain = DatasetGenerator( pathDatasetFile=pathFileTrain,transCrop=transCrop)
        datasetVal = DatasetGenerator( pathDatasetFile=pathFileVal,transCrop=transCrop)

        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True, num_workers=10,
                                     pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=10,
                                   pin_memory=True)

        optimizer = optim.Adam([
            {'params': model.module.paramgroup01(), 'lr': 1e-6},
            {'params': model.module.paramgroup234(), 'lr': 1e-4},
            {'params': model.module.parameters_classifier(), 'lr': 1e-2},
        ])

        def schedule(epoch):
            if epoch < 2:
                ub = 1
            elif epoch < 4:
                ub = 0.1
            else:
                ub = 0.01
            return ub

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda e: schedule(e),
                                                                            lambda e: schedule(e),
                                                                            lambda e: schedule(e)])

        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        lossMIN = 100000

        currentacc = 0
        for epochID in range(0, trMaxEpoch):

            Trainer.epochTrain(model, dataLoaderTrain, optimizer, loss)
            lossVal2, losstensor2, accuracy2, precision_class2, recall_class2 = Trainer.epochVal(model,
                                                                                                 dataLoaderTrain,
                                                                                                 nnClassCount, loss)

            lossVal, losstensor, accuracy, precision_class, recall_class = Trainer.epochVal(model, dataLoaderVal,
                                                                                            nnClassCount, loss)

            timestampTime = time.strftime("%H:%M:%S")
            timestampDate = time.strftime("%Y-%m-%d")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step()

            precision_class = list(precision_class * 100)
            precision_class2 = list(precision_class2 * 100)
            recall_class = list(recall_class * 100)
            recall_class2 = list(recall_class2 * 100)

            if lossVal < lossMIN:
                currentacc = accuracy

                lossMIN = lossVal
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()},
                           './models/m-' + launchTimestamp + '.pth.tar')
                print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] lossTrain= ' + str(
                    lossVal2) + ' lossVal= ' + str(
                    lossVal) + ' Accuracy:' + str(accuracy2)[0:5] + '% & ' + str(accuracy)[0:5] + '%')
                print('training set:')
                for i in range(0, nnClassCount):
                    print('\'' + str(i) + '\'class precision：' + str(precision_class2[i])[0:5] + '%')
                    print('\'' + str(i) + '\'class recall：' + str(recall_class2[i])[0:5] + '%')
                print('validating set:')
                for i in range(0, nnClassCount):
                    print('\'' + str(i) + '\'class precision：' + str(precision_class[i])[0:5] + '%')
                    print('\'' + str(i) + '\'class recall：' + str(recall_class[i])[0:5] + '%')
            else:
                print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] lossTrain= ' + str(
                    lossVal2) + ' lossVal= ' + str(
                    lossVal) + ' Accuracy:' + str(accuracy2)[0:5] + '% & ' + str(accuracy)[0:5] + '%')
                print('training set:')
                for i in range(0, nnClassCount):
                    print('\'' + str(i) + '\'class precision：' + str(precision_class2[i])[0:5] + '%')
                    print('\'' + str(i) + '\'class recall：' + str(recall_class2[i])[0:5] + '%')
                print('validating set:')
                for i in range(0, nnClassCount):
                    print('\'' + str(i) + '\'class precision：' + str(precision_class[i])[0:5] + '%')
                    print('\'' + str(i) + '\'class recall：' + str(recall_class[i])[0:5] + '%')

        os.rename('./models/m-' + launchTimestamp + '.pth.tar',
                  './models/m-' + str(currentacc) + '-' + launchTimestamp + '.pth.tar')

    def epochTrain(model, dataLoader, optimizer, loss):

        model.train()

        for batchID, (input, target) in enumerate(dataLoader):
            target = target.cuda()

            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)
            varOutput = model(varInput)

            lossvalue = loss(varOutput, varTarget)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

    # --------------------------------------------------------------------------------

    def epochVal(model, dataLoader, classCount, loss):

        model.eval()

        lossVal = 0
        lossValNorm = 0

        losstensorMean = 0
        accMean = 0
        acc_class_mean = [0 for i in range(0, classCount)]
        acc_class_mean2 = [0 for i in range(0, classCount)]
        acc_class_count = [0 for i in range(0, classCount)]
        acc_class_count2 = [0 for i in range(0, classCount)]
        accClass = []
        accClass2 = []

        for i, (input, target) in enumerate(dataLoader):
            target = target.cuda()

            with torch.no_grad():
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)
                varOutput = model(varInput)

            losstensor = loss(varOutput, varTarget)
            acc = Trainer.ac(varOutput, varTarget)
            losstensorMean += losstensor
            accMean += acc
            accclass = []
            accclass2 = []

            for c in range(0, classCount):
                accclass.append(Trainer.ac_precision(c, varOutput, varTarget))
                accclass2.append(Trainer.ac_recall(c, varOutput, varTarget))

            lossVal += losstensor.item()
            lossValNorm += 1
            accClass.append(accclass)
            accClass2.append(accclass2)

        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm
        accMean = accMean / lossValNorm
        for c in range(0, classCount):
            for ii in range(0, lossValNorm):
                if accClass[ii][c] != 2:
                    acc_class_mean[c] += accClass[ii][c]
                    acc_class_count[c] += 1
                if accClass2[ii][c] != 2:
                    acc_class_mean2[c] += accClass2[ii][c]
                    acc_class_count2[c] += 1
        acc_class_mean = np.array(acc_class_mean) / np.array(acc_class_count)
        acc_class_mean2 = np.array(acc_class_mean2) / np.array(acc_class_count2)

        return outLoss, losstensorMean, accMean, acc_class_mean, acc_class_mean2

    def epochTest(model, dataLoader, classCount, loss, excel_path):

        model.eval()

        lossVal = 0
        lossValNorm = 0

        losstensorMean = 0
        accMean = 0
        acc_class_mean = [0 for i in range(0, classCount)]
        acc_class_mean2 = [0 for i in range(0, classCount)]
        acc_class_count = [0 for i in range(0, classCount)]
        acc_class_count2 = [0 for i in range(0, classCount)]
        accClass = []
        accClass2 = []

        for i, (input, target) in enumerate(dataLoader):
            target = target.cuda()

            with torch.no_grad():
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)
                varOutput = model(varInput)

            losstensor = loss(varOutput, varTarget)
            acc = Trainer.ac(varOutput, varTarget)
            Trainer.result_to_excel(varOutput, varTarget, excel_path)
            losstensorMean += losstensor
            accMean += acc
            accclass = []
            accclass2 = []

            for c in range(0, classCount):
                accclass.append(Trainer.ac_precision(c, varOutput, varTarget))
                accclass2.append(Trainer.ac_recall(c, varOutput, varTarget))

            lossVal += losstensor.item()
            lossValNorm += 1
            accClass.append(accclass)
            accClass2.append(accclass2)

        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm
        accMean = accMean / lossValNorm
        for c in range(0, classCount):
            for ii in range(0, lossValNorm):
                if accClass[ii][c] != 2:
                    acc_class_mean[c] += accClass[ii][c]
                    acc_class_count[c] += 1
                if accClass2[ii][c] != 2:
                    acc_class_mean2[c] += accClass2[ii][c]
                    acc_class_count2[c] += 1
        acc_class_mean = np.array(acc_class_mean) / np.array(acc_class_count)
        acc_class_mean2 = np.array(acc_class_mean2) / np.array(acc_class_count2)

        return outLoss, losstensorMean, accMean, acc_class_mean, acc_class_mean2

    def ac(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return (torch.tensor(torch.sum(preds == labels).item() / len(preds))).item() * 100

    def ac_precision(clas, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        _labels = labels.cpu().numpy()
        _preds = preds.cpu().numpy()
        tc = 0
        t_c = 0
        for i in range(0, len(_labels)):
            if _preds[i] == clas:
                tc += 1
                if _labels[i] == clas:
                    t_c += 1
        if tc != 0:
            return t_c / tc
        else:
            return 2

    def ac_recall(clas, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        _labels = labels.cpu().numpy()
        _preds = preds.cpu().numpy()
        tc = 0
        t_c = 0
        for i in range(0, len(_labels)):
            if _labels[i] == clas:
                tc += 1
                if _preds[i] == clas:
                    t_c += 1
        if tc != 0:
            return t_c / tc
        else:
            return 2

    def result_to_excel(outputs, labels, excel_path):
        _outputs = outputs.cpu().numpy()
        _, preds = torch.max(outputs, dim=1)
        _labels = labels.cpu().numpy()
        _preds = preds.cpu().numpy()

        exl = op.load_workbook(excel_path)
        ws = exl["Sheet1"]

        for i in range(0, len(_labels)):
            ws.append([_labels[i], _preds[i], _outputs[i][0], _outputs[i][1]])

        exl.save(excel_path)

        return 0

    def drawROC(y, scores, timestampDate):
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)
        auc = metrics.auc(fpr, tpr)
        print(auc)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 label='ROC curve (area = %0.4f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('./Results/{}.png'.format(timestampDate))
        plt.show()

    def test(pathDirData, pathFileTest, pathModel, nnClassCount,
             trBatchSize, transCrop, excel_path):

        cudnn.benchmark = True
        model = get_model()

        model = torch.nn.DataParallel(model).cuda()

        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        transformList = [transforms.Resize([transCrop, transCrop]),
                         transforms.ToTensor()]

        transformSequence = transforms.Compose(transformList)

        datasetTest = DatasetGenerator( pathDatasetFile=pathFileTest,
                                       transCrop=transCrop)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=0, shuffle=False,
                                    pin_memory=True)

        loss = torch.nn.CrossEntropyLoss(reduction='mean')
        lossVal, losstensor, accuracy, precision_class, recall_class = Trainer.epochTest(model, dataLoaderTest,
                                                                                         nnClassCount,
                                                                                         loss,
                                                                                         excel_path)

        precision_class = list(precision_class * 100)
        recall_class = list(recall_class * 100)

        print('validating set:' + ' Accuracy:' + str(accuracy)[0:5] + '%')
        for i in range(0, nnClassCount):
            print('\'' + str(i) + '\'class precision：' + str(precision_class[i])[0:5] + '%')
            print('\'' + str(i) + '\'class recall：' + str(recall_class[i])[0:5] + '%')

        exl = xlrd.open_workbook(excel_path)
        sheet = exl.sheet_by_name("Sheet1")
        labelClass = sheet.col_values(0)
        predClass = sheet.col_values(3)

        timestampDate = time.strftime("%Y%m%d")
        timestampDate = timestampDate[2:]
        Trainer.drawROC(labelClass, predClass, timestampDate)

        return


