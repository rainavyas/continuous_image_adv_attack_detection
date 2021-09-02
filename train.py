'''
Main training script
'''

import torch
import torchvision
from torchvision import models
import torch.nn as nn
from cnn_finetune import make_model
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tools import get_default_device, accuracy_topk, AverageMeter
import sys
import os
import argparse

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=50):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):


        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy_topk(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t Loss {loss.val:.4f} ({loss.avg:.4f}) \t Prec@1 {top1.val:.3f} ({top1.avg:.3f})')
    
def validate(val_loader, model, criterion, device):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy_topk(output.data, target)
            prec5 = accuracy_topk(output.data, target, 5)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    print('Test\t  Prec@1: {top1.avg:.3f} (Err: {error:.3f} )\n'
          .format(top1=top1,error=100-top1.avg))
    print('Test\t  Prec@5: {top5.avg:.3f} (Err: {error:.3f} )\n'
          .format(top5=top5,error=100-top5.avg))

    return top1.avg, top5.avg


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify output th file')
    commandLineParser.add_argument('ARCH', type=str, help='vgg16')
    commandLineParser.add_argument('--B', type=int, default=128, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=100, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.001, help="Specify learning rate")
    commandLineParser.add_argument('--weight', type=float, default=1e-4, help="Specify weight decay")
    commandLineParser.add_argument('--momentum', type=float, default=0.9, help="Specify momentum")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")

    args = commandLineParser.parse_args()
    out_file = args.OUT
    arch = args.ARCH
    batch_size = args.B
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight
    momentum = args.momentum
    seed = args.seed

    torch.manual_seed(seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the data
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2009, 0.1984, 0.2023], 
        ),
    ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2009, 0.1984, 0.2023], 
        ),
    ])),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

    # Initialse model using ImageNet pre-trained model
    model = make_model(arch, num_classes=100, pretrained=True, input_size=(32,32))
    model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150])

    # training loop
    for epoch in range(epochs):

        # train for one epoch
        print(f'Training {arch} model')
        print(f'Current lr {lr:.5e}')
        train(train_loader, model, criterion, optimizer, epoch, device)

        # Evaluate on validation set
        _, _ = validate(test_loader, model, criterion, device)
    
    # Save the trained model
    state = model.state_dict()
    torch.save(state, out_file)

