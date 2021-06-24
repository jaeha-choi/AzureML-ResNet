import argparse
import logging

import torch
import torchmetrics
import torch.nn as nn
import torch.optim as opt
# Azure
import torchvision
from torchvision import transforms
from azureml.core import Run
from torch.utils.data import DataLoader

from src.resnet50_15 import Resnet50v15, Resnet50v15Classifier
from util.tiny_imagenet import TinyImagenet
from util.tiny_imagenet_val import TinyImagenetVal


def main():
    run = Run.get_context()
    # Azure end

    # ------- Model parameters ------- #
    parser = argparse.ArgumentParser("Resnet50v15")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs", default=70)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-2)
    parser.add_argument("--batch", type=int, help="Size of each batch", default=32)
    parser.add_argument("--shuffle", type=bool, help="Whether to shuffle the data", default=True)
    parser.add_argument("--dataloc", type=str, help="TinyImagenet200 dataset location", default="./dataset")
    parser.add_argument("--output_dir", type=str, help="Directory location to save the model outputs", default="./out")
    args = parser.parse_args()

    EPOCH = args.num_epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch  # number of batches (SAMPLE_SIZE / BATCH_SIZE per batch)
    SHUFFLE_DATA = args.shuffle
    DATASET_LOCATION = args.dataloc
    OUTPUT_DIR = args.output_dir
    IMAGE_SIZE = 64

    # Basic logger setting
    log = logging.getLogger("train_resnet")
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", '%m/%d/%Y %I:%M:%S %p')
    stream_h = logging.StreamHandler()
    stream_h.setLevel(logging.DEBUG)
    stream_h.setFormatter(formatter)
    log.addHandler(stream_h)

    # Use gpu if possible
    device_cnt = torch.cuda.device_count()
    log.info("Number of available GPUs: %s" % device_cnt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Current device: %s" % device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    print(len(trainset))

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    log.info("Dataset is ready.")

    # Init model
    log.info("Preparing model...")
    resnet = torchvision.models.resnet50()
    resnet = resnet.to(device)

    log.info("Model is ready.")

    # Init params
    log.info("Setting hyperparameters...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = opt.SGD(resnet.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    log.info("Ready for training.")

    for epoch in range(EPOCH):
        log.info("Epoch: %s" % (epoch + 1))
        running_loss = 0
        running_corrects = torch.zeros(1).to(device)
        # Set to training mode
        resnet.train()
        for i, (img, label) in enumerate(trainloader):
            # Train models here
            optimizer.zero_grad()  # gradient reset
            img = img.to(device)
            label = label.to(device)
            pred = resnet(img)
            loss = loss_fn(pred, label)  # forward prop
            # print("label", label)
            loss.backward()  # backward prop
            optimizer.step()  # update gradients
            scheduler.step(loss)
            # calculate metrics
            running_loss = loss.item()
            # torch.max gives values and indices
            _, pred_idx = torch.max(pred, 1)
            running_corrects += torch.eq(pred_idx, label).sum()
            # scheduler.step(val_loss) # note that scheduler must be used after the training steps

            if (i + 1) % 100 == 0:
                log.info("Training:: Epoch: %s/%s\tBatch: %s/%s\t\tLoss: %.8f" % (
                    epoch + 1, EPOCH, i + 1, len(trainloader), running_loss))
        # Azure
        run.log('train_loss', running_loss)
        epoch_acc = (running_corrects.double() / len(trainset)).item()
        run.log('train_acc', epoch_acc)
        # Azure end

        running_loss_val = 0
        running_corrects_val = torch.zeros(1).to(device)
        # Set to validation mode
        resnet.eval()
        with torch.no_grad():
            for i, (img, label) in enumerate(testloader):
                img = img.to(device)
                label = label.to(device)
                pred = resnet(img)
                loss = loss_fn(pred, label)

                # calculate metrics
                running_loss_val = loss.item()
                _, pred_idx = torch.max(pred, 1)
                running_corrects_val += torch.eq(pred_idx, label).sum()
                if (i + 1) % 10 == 0:
                    log.info("Validation:: Epoch: %s/%s\tBatch: %s/%s\t\tLoss: %s" % (
                        epoch + 1, EPOCH, i + 1, len(testloader), running_loss_val))
        # Azure
        run.log('val_loss', running_loss_val)
        epoch_acc_val = (running_corrects_val.double() / len(testset)).item()
        run.log('val_acc', epoch_acc_val)
        # Azure end


if __name__ == '__main__':
    main()
