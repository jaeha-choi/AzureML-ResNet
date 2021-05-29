import logging as log

import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, Subset

from src.resnet50_15 import Resnet50v15, Resnet50v15Classifier
from util.tiny_imagenet import TinyImagenet

log.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                level=log.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')

# ------- Model parameters ------- #
EPOCH = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 2 # number of batches (SAMPLE_SIZE / BATCH_SIZE per batch)
SHUFFLE_DATA = False
DATASET_LOCATION = "../tiny-imagenet-200"

dataset = TinyImagenet(DATASET_LOCATION)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)

# image = torch.randn(8, 3, 224, 224)

resnet = Resnet50v15()
resnet_classifier = Resnet50v15Classifier(resnet, classes=200, softmax=False)
#print(resnet_classifier)

loss_fn = nn.CrossEntropyLoss()
optimizer = opt.SGD(resnet_classifier.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

for epoch in range(EPOCH):
    log.info("Epoch: %s" % (epoch + 1))
    running_loss = 0
    for i, (img, label) in enumerate(train_dataloader):
        # Train models here
        optimizer.zero_grad() # gradient reset
        pred = resnet_classifier(img)
        loss = loss_fn(pred, label) # forward prop
        print("label", label)
        loss.backward() # backward prop
        optimizer.step() # update gradients
        
        #scheduler.step(val_loss) # note that scheduler must be used after the training steps
        running_loss = loss.item()
        if True: #(i + 1) % 10 == 0:
            log.info("Epoch: %s\tBatch: %s\tTrainLoss: %s" % (epoch + 1, i + 1, running_loss))
        # break
    # break