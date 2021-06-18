import argparse
import logging

import torch
import torchmetrics
import torch.nn as nn
import torch.optim as opt
# Azure
import torchvision
from azureml.core import Run
from torch.utils.data import DataLoader

from src.resnet50_15 import Resnet50v15, Resnet50v15Classifier
from util.tiny_imagenet import TinyImagenet
from util.tiny_imagenet_val import TinyImagenetVal

run = Run.get_context()
# Azure end

# ------- Model parameters ------- #
parser = argparse.ArgumentParser("Resnet50v15")
parser.add_argument("--num_epochs", type=int, help="Number of epochs", default=32)
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
parser.add_argument("--batch", type=int, help="Size of each batch", default=2)
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

# Load datasets
log.info("Loading dataset from " + DATASET_LOCATION)
train_dataset = TinyImagenet(DATASET_LOCATION, img_crop_size=IMAGE_SIZE, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.CenterCrop(IMAGE_SIZE)
]))
# train_dataset.save_all_dict() # This line is disabled for read-only file system
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)
val_dataset = TinyImagenetVal(DATASET_LOCATION, img_crop_size=IMAGE_SIZE, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.CenterCrop(IMAGE_SIZE)
]))
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
log.info("Dataset is ready.")

# Init model
log.info("Preparing model...")
resnet = Resnet50v15()
resnet = resnet.to(device)
resnet_classifier = Resnet50v15Classifier(resnet, classes=200, softmax=False)
resnet_classifier = resnet_classifier.to(device)

resnet_classifier_inference = nn.Sequential(
    resnet_classifier,
    nn.Softmax(dim=-1),
)
resnet_classifier_inference = resnet_classifier_inference.to(device)
# print(resnet_classifier)

metric_acc = torchmetrics.Accuracy().to(device)
log.info("Model is ready.")

# Init params
log.info("Setting hyperparameters...")
loss_fn = nn.CrossEntropyLoss()
optimizer = opt.SGD(resnet_classifier.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
log.info("Ready for training.")

for epoch in range(EPOCH):
    log.info("Epoch: %s" % (epoch + 1))
    running_loss = 0
    running_corrects = torch.zeros(1).to(device)
    # Set to training mode
    resnet.train()
    for i, (img, label) in enumerate(train_dataloader):
        # Train models here
        optimizer.zero_grad()  # gradient reset
        img = img.to(device)
        label = label.to(device)
        pred = resnet_classifier(img)
        loss = loss_fn(pred, label)  # forward prop
        # print("label", label)
        loss.backward()  # backward prop
        optimizer.step()  # update gradients

        # calculate metrics
        running_loss = loss.item()
        # torch.max gives values and indices
        _, pred_idx = torch.max(pred, 1)
        running_corrects += torch.eq(pred_idx, label).sum()
        # scheduler.step(val_loss) # note that scheduler must be used after the training steps

        if (i + 1) % 100 == 0:
            log.info("Training:: Epoch: %s/%s\tBatch: %s/%s\t\tLoss: %.8f" % (
                epoch + 1, EPOCH, i + 1, len(train_dataloader), running_loss))
    # Azure
    run.log('train_loss', running_loss)
    epoch_acc = (running_corrects.double() / len(train_dataset)).item()
    run.log('train_acc', epoch_acc)
    # Azure end

    running_loss_val = 0
    running_corrects_val = torch.zeros(1).to(device)
    # Set to validation mode
    resnet.eval()
    with torch.no_grad():
        for i, (img, label) in enumerate(val_dataloader):
            img = img.to(device)
            label = label.to(device)
            pred = resnet_classifier(img)
            loss = loss_fn(pred, label)

            # calculate metrics
            running_loss_val = loss.item()
            _, pred_idx = torch.max(pred, 1)
            running_corrects_val += torch.eq(pred_idx, label).sum()
            if (i + 1) % 10 == 0:
                log.info("Validation:: Epoch: %s/%s\tBatch: %s/%s\t\tLoss: %s" % (
                    epoch + 1, EPOCH, i + 1, len(train_dataloader), running_loss_val))
    # Azure
    run.log('val_loss', running_loss_val)
    epoch_acc_val = (running_corrects_val.double() / len(val_dataset)).item()
    run.log('val_acc', epoch_acc_val)
    # Azure end
