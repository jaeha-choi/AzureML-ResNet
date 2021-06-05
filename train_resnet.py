import os
import sys
import argparse

import logging as log

import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, Subset

from src.resnet50_15 import Resnet50v15, Resnet50v15Classifier
from util.tiny_imagenet import TinyImagenet

# Azure
from azureml.core import Run
run = Run.get_context()
# Azure end

log.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                level=log.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
log.info("Logger:", str(log));

# ------- Model parameters ------- #
parser = argparse.ArgumentParser("Resnet50v15")
parser.add_argument("--num_epochs", type=int, help="Number of epochs", default=32)
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
parser.add_argument("--batch", type=int, help="Size of each batch", default=2)
parser.add_argument("--shuffle", type=bool, help="Whether to shuffle the data", default=True)
parser.add_argument("--dataloc", type=str, help="TinyImagenet200 dataset location")
parser.add_argument("--output_dir", type=str, help="Directory location to save the model outputs")
args = parser.parse_args()

EPOCH = args.num_epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch # number of batches (SAMPLE_SIZE / BATCH_SIZE per batch)
SHUFFLE_DATA = args.shuffle
DATASET_LOCATION = args.dataloc
OUTPUT_DIR = args.output_dir

log.info("Loading dataset from "+DATASET_LOCATION)
dataset = TinyImagenet(DATASET_LOCATION)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)
log.info("Dataset is ready.")

# image = torch.randn(8, 3, 224, 224)

log.info("Preparing model...")
resnet = Resnet50v15()
resnet_classifier = Resnet50v15Classifier(resnet, classes=200, softmax=False)
#print(resnet_classifier)
log.info("Model is ready.")

log.info("Setting hyperparameters...")
loss_fn = nn.CrossEntropyLoss()
optimizer = opt.SGD(resnet_classifier.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
log.info("Ready for training.")

for epoch in range(EPOCH):
    log.info("Epoch: %s" % (epoch + 1))
    running_loss = 0
    for i, (img, label) in enumerate(train_dataloader):
        # Train models here
        optimizer.zero_grad() # gradient reset
        pred = resnet_classifier(img)
        loss = loss_fn(pred, label) # forward prop
        # print("label", label)
        loss.backward() # backward prop
        optimizer.step() # update gradients
        
        #scheduler.step(val_loss) # note that scheduler must be used after the training steps
        running_loss = loss.item()
        if True: #(i + 1) % 10 == 0:
            log.info("Epoch: %s\tBatch: %s\tTrainLoss: %s" % (epoch + 1, i + 1, running_loss))
            
            # Azure
            run.log('train_loss', running_loss)
            # Azure end
        # break
    # break
