import torch
import logging
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil
import os
from torch.utils.data import TensorDataset, DataLoader
from PairwiseContrastive import PairwiseContrastive
from F1_Loss import F1_Loss
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from Focal_Loss import FocalLoss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
logger = logging.getLogger(__name__)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



def train(model, train_loader, criterion, criterion_ms, optimizer, ms_weight, cross_entropy_weight,criterion_f1):
    model.train()
    global  best_f1
    y_preds = []
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    pbar = tqdm(train_loader)
    for i_batch, sample_batched in enumerate(train_loader):
        x_train = sample_batched[0]

        y_train = sample_batched[1]

        features, pred = model(x_train)
        _, pred_out = torch.max(pred.data, 1)
        predlist = torch.cat([predlist, pred_out.view(-1).cpu()])
        lbllist = torch.cat([lbllist, y_train.view(-1).cpu()])

        Lx = F.cross_entropy(pred, y_train, reduction='mean')

        loss_ms = criterion_ms(features, y_train)

        loss = Lx + ms_weight * loss_ms
        pbar.set_description(
            "loss_ms: {loss_ms:.4f}. loss: {loss:.4f}s.cross_entropy: {cross_entropy:.4f}s. best_f1: {best_f1:.4f}s.".format(
                loss_ms=loss_ms,
                loss=loss,
                cross_entropy=Lx,
                best_f1 =best_f1
                ))
        pbar.update()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return lbllist, predlist


def test(model, test_loader):
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    outlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    model.eval()
    outputs = []
    for i_batch, sample_batched in enumerate(test_loader):
        x_train = sample_batched[0]

        y_train = sample_batched[1]
        feautes, output = model(x_train)
        _, output = torch.max(output.data, 1)

        outlist = torch.cat([outlist, output.view(-1).cpu()])
        lbllist = torch.cat([lbllist, y_train.view(-1).cpu()])

    return outlist, lbllist


def loaddata():
    X_train = np.load('/home/path')
    Y_train = np.load('/home/path').astype(np.int64)
    X_test= np.load('/home/path')
    Y_test= np.load('/home/path').astype(np.int64)
    X_train = torch.FloatTensor(X_train).cuda()
    Y_train = torch.LongTensor(Y_train).cuda()
    X_test = torch.FloatTensor(X_test).cuda()
    Y_test = torch.LongTensor(Y_test).cuda()
    trainSet = TensorDataset(X_train, Y_train)
    testSet = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(trainSet, batch_size=32,drop_last=True,shuffle=True)
    test_loader = DataLoader(testSet, batch_size=32)
    return train_loader, test_loader


class TabularModel(nn.Module):

    def __init__(self, n_cont, out_sz, out_dim, layers, p=0.5):
        # Call the parent __init__
        super().__init__()

        # Set up the embedding, dropout, and batch normalization layer attributes
        #         self.bn_cont = nn.BatchNorm1d(n_cont)

        # Assign a variable to hold a list of layers
        layerlist = []

        n_in = n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU())
            #             layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))

        # Convert the list of layers into an attribute
        self.layers = nn.Sequential(*layerlist)
        self.fc = nn.Linear(out_sz, out_dim)

    def forward(self, x_cont):

        features = self.layers(x_cont)
        pred = self.fc(features)
        return features, pred


def cmd_argu():
    parser = argparse.ArgumentParser(description='PyTorch MultiCon Training')
    parser.add_argument('--gpuid', default='2', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--epochs', default=250, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--dataroot', type=str, default='/home/path',
                        help='The path of processed drug data, must be available when dataset is drug')
    parser.add_argument('--eval', type=str, default='no data', help='The path of test dataset')
    parser.add_argument('--nnstructure', type=str, help='Neural Network hidden layer size')
    parser.add_argument('--embeding', type=int, default=512, help='embeding')
    parser.add_argument('--imputdim', type=int, help='imput dim')
    parser.add_argument('--dropout', type=float, default=0.02, help='dropout')
    parser.add_argument('--numworkers', type=int, default=4, help='number of workers')
    parser.add_argument('--numclass', type=int, default=2, help='numclass')
    parser.add_argument('--msweight', type=float, default=0.8, help='msweight')
    parser.add_argument('--crossentropyweight', type=int, default=1, help='crossentropyweight')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--test', default='', type=bool, help='Ture just test (default: none)')

    return parser


def main():
    parser = cmd_argu()
    args = parser.parse_args()
    epochs = args.epochs
    device = args.gpuid
    lr = args.lr



    layers = args.nnstructure.split(',')
    layers = [int(s) for s in layers]
    ms_weight = args.msweight
    crossentropyweight = args.crossentropyweight

    embeding = args.embeding
    input_dim = args.imputdim
    classes = args.numclass
    ms_weight = args.msweight
    trainloader, testloader = loaddata()
    p = args.dropout
    # print("data root", dataroot)
    print("total epochs:", epochs)
    print("input_dim", input_dim)
    print("layers:", layers)
    print("embeding", embeding)
    print("device", device)
    print("ms_weight", ms_weight)
    print("crossentropyweight", crossentropyweight)
    model = TabularModel(n_cont=input_dim, out_sz=embeding, out_dim=classes, layers=layers, p=p).cuda()


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, nesterov=True)
    criterion_ms = PairwiseContrastive().cuda()
    criterion_f1 = F1_Loss().cuda()
    global best_f1
    global best_epoch
    global best_y_pred
    global best_lbllist
    start_epoch=0
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_f1 = checkpoint['best_f1']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.test == True:
        with torch.no_grad():
            y_pred, lbllist = test(model, testloader)

            conf_mat = confusion_matrix(lbllist, y_pred)

            logger.info(confusion_matrix(y_pred, lbllist))
            acc = accuracy_score(lbllist, y_pred)



    model.zero_grad()
    print(model)
    for i in range(start_epoch,epochs):
        i += 1

        y_pred, y_label = train(model, trainloader, criterion=criterion, criterion_ms=criterion_ms, optimizer=optimizer,
                               ms_weight=ms_weight, cross_entropy_weight=crossentropyweight,criterion_f1=criterion_f1)


        if i % 100 == 0:
            with torch.no_grad():
                y_pred, lbllist = test(model, testloader)

                logger.info(confusion_matrix(y_pred, lbllist))


                model_to_save = model
                save_checkpoint({
                    'epoch': i + 1,
                    'state_dict': model_to_save.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': optimizer.state_dict(),
                })

                print("best_f1", best_f1, "best epoch", best_epoch)

        if i == epochs - 1:
            with torch.no_grad():
                y_pred, lbllist = test(model, testloader)

                classification_report(lbllist, y_pred, labels=[0, 1])

                acc = accuracy_score(lbllist, y_pred)

                f1 = f1_score(lbllist, y_pred, average='binary')




if __name__ == '__main__':
    main()





