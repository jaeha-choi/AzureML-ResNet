import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader

from src.resnet50_15 import Resnet50v15, Resnet50v15Classifier
from util.tiny_imagenet import TinyImagenet

# ------- Model parameters ------- #
EPOCH = 5
LEARNING_RATE = 0.1
BATCH_SIZE = 3
SHUFFLE_DATA = True

dataset = TinyImagenet()
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)

# image = torch.randn(8, 3, 224, 224)

resnet = Resnet50v15()
resnet_classifier = Resnet50v15Classifier(resnet, 200)

loss_fn = nn.CrossEntropyLoss()
optimizer = opt.SGD(resnet_classifier.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

for epoch in range(EPOCH):
    print("\nEpoch: %s" % epoch)
    running_loss = 0
    for i, (img, label) in enumerate(train_dataloader):
        # Train models here
        print("1")
        optimizer.zero_grad()
        loss = loss_fn(resnet_classifier(img), label)
        loss.backward()
        print("2")
        scheduler.step(loss)
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print("Epoch: %s\tBatch: %s\tLoss: %s" % (epoch + 1, i + 1, running_loss))
        # break
    break
