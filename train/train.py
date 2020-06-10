from __future__ import print_function

import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys

from model import arch_model
from prepare_dataset import prepare_dataset


num_classes = 101

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 123
threads = 8
lr = 0.0001
nEpoch = 1

batchSize = 40
testBatchSize = 40

torch.manual_seed(seed)

modelA = models.resnet101(pretrained=True, progress=True).to(device)
modelB = models.resnet34(pretrained=True, progress=True).to(device)
#modelC = models.resnet152(pretrained=True, progress=True).to(device)
#modelB = models.resnet101(pretrained=True, progress=True).to(device)

for param in modelA.parameters():
    param.requires_grad = False
for param in modelB.parameters():
    param.requires_grad = False

modelA.avgpool = arch_model.ConvNet_RN101().to(device)
modelB.avgpool = arch_model.ConvNet_RN34().to(device)

model = arch_model.MyEnsemble(modelA, modelB, num_classes).to(device)

summary(model, input_size=(3, 224, 224))

#####################################################################################################################

criterion = nn.BCELoss().to(device)
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
optimizer = optim.Adam(model.parameters(), lr=lr)

#####################################################################################################################
train_set = prepare_dataset.get_training_set(num_classes)
test_set = prepare_dataset.get_test_set(num_classes)

training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchSize,
                                          shuffle=True)

testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=batchSize,
                                         shuffle=True)
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)


def train(epoch):
    epoch_loss = 0
    epoch_loss_mse = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        result = model(input)

        #loss_vgg_r = loss_network(result).to(device)
        #loss_vgg_t = loss_network(target).to(device)

        loss = criterion(result, target)# + criterion(loss_vgg_r, loss_vgg_t) * 1e-3
        epoch_loss += loss.item()
        #loss_mse = criterion(result, target)
        #epoch_loss_mse += loss_mse.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(scheduler.get_lr())

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss.item()))

    loss_per_epoch = epoch_loss / len(training_data_loader)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, loss_per_epoch))
    return loss_per_epoch


def test(epoch):
    epoch_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for iteration, batch in enumerate(testing_data_loader, 1):
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            loss = criterion(prediction, target)
            epoch_loss += loss.item()


            _, predicted = torch.max(prediction.data, 1)
            _, target_idx = torch.max(target.data, 1)

            total += target.size(0)
            correct += (predicted == target_idx).sum().item()

    loss_per_epoch = epoch_loss / len(testing_data_loader)
    acc = 100 * correct / total
    print("===> Avg. Accuracy: {:.4f} %".format(acc))
    print("===> Avg. Loss: {:.4f}".format(loss_per_epoch))
    return acc, loss_per_epoch


def train_test():
    best_loss = sys.float_info.max
    best_loss_valid = sys.float_info.max
    list_loss = []
    list_loss_val = []
    accuracy = []
    for epoch in range(1, nEpoch + 1):
        loss_per_epoch = train(epoch)
        list_loss.append(loss_per_epoch)
        acc, loss_valid = test(epoch)
        list_loss_val.append(loss_valid)
        accuracy.append(acc)
        if loss_per_epoch < best_loss and loss_valid < best_loss_valid:
            best_loss = loss_per_epoch
            best_loss_valid = loss_valid
            model_out_path = "pretrain_models/ResNet101_test.pth"
            torch.save(model.state_dict(), model_out_path)
            print("Checkpoint saved to {}".format(model_out_path))

    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ep = np.arange(1, nEpoch + 1, 1)
    ax1.plot(ep, list_loss)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train Data')
    fig1.tight_layout()
    ax1.grid()
    plt.show()

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    ep = np.arange(1, nEpoch + 1, 1)
    ax2.plot(ep, list_loss_val)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Test Data')
    fig2.tight_layout()
    ax2.grid()
    plt.show()

    fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
    ep = np.arange(1, nEpoch + 1, 1)
    ax3.plot(ep, accuracy)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Test Data')
    fig3.tight_layout()
    ax3.grid()
    plt.show()