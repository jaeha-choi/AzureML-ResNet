# AzureML-ResNet

### Directory structure

```
AzureML-ResNet/
├── dataset
│   ├── json
│   ├── test
│   ├── train
│   ├── val
│   ├── wnids.txt
│   └── words.txt
├── example
│   ├── TinyImagenet.ipynb
│   └── train_resnet.ipynb
├── src
│   └── resnet50_15.py
├── testing
│   ├── resblock_testing.ipynb
│   └── resnet_debug.py
└── util
    ├── tiny_imagenet.py
    └── tiny_imagenet_val.py
```

- `dataset`: Contains extracted Tiny Imagenet dataset: http://cs231n.stanford.edu/tiny-imagenet-200.zip
- `example`: Contains executable example files.
- `src`: Contains source files that will be uploaded on Azure.
- `testing`: Contains testing files. (to be removed)
- `util`: Contains utility classes such as dataset classes.

