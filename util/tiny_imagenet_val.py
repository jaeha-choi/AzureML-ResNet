import json
import os

import PIL.Image
import numpy as np
import torch.utils.data
import torchvision.transforms as vision
import random
# import matplotlib.pyplot as plt

class TinyImagenetVal(torch.utils.data.Dataset):

    def __init__(self, base_dir: str = "./dataset/", src_dir: str = "val", transform=None, img_crop_size: int = 224):
        """
        Initializes validation dataset for tiny imagenet.
        Requires id2int.json and int2name.json file in the base directory.
        :param base_dir: Dataset directory. Contains test/, train/, val/, words.txt, etc
        :param src_dir: Source directory for validation data. E.g: val
        :param transform: A function to use for transforming images
        :param img_crop_size: Target image size. If image size is smaller, paddings will be added.
        """
        # Dataset directory. Contains test/, train/, val/, words.txt, etc
        self._base_dir = base_dir
        self._src_dir = os.path.join(base_dir, src_dir)

        self._images = []
        self._id2int = {}  # n02056570 -> 2
        self._int2name = {}  # 2 -> "king penguin, Aptenodytes patagonica"

        self._img_size = img_crop_size

        # If some transformation is necessary, create a function for it
        self.transform = transform

        # Load index (label) files that were created in tiny_imagenet.py
        self._load_saved_files()
        # Read image file names from the source directory
        self._load_images()

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx) -> (torch.Tensor, int):
        """
        Reads the RGB image file at idx, and return it as an image tensor with an int label.
        Int label can be converted to a human-readable label with get_class_name() function.
        :param idx: Index of data to get
        :return: Tensor containing image as an array, and an integer representing the image number
        """
        img_name = self._images[idx][0]
        img_label = self._images[idx][1]
        full_file_name = os.path.join(self._src_dir, "images", img_name)

        image = PIL.Image.open(full_file_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        cropped = vision.CenterCrop(self._img_size)
        arr = np.transpose(np.array(cropped(image)), (2, 0, 1))
        return torch.from_numpy(arr).float(), img_label

    def get_class_name(self, int_label) -> str:
        """
        Convert an integer label to a human-readable class name
        :param int_label: Integer label to convert
        :return: Human-readable class name
        """
        return self._int2name[int_label]

    def _load_images(self):
        with open(os.path.join(self._src_dir, "val_annotations.txt"), "r", encoding="utf-8") as f:
            for line in f:
                ln = line.strip().split()
                img_name = ln[0]
                img_id = int(self._id2int[ln[1]])
                self._images.append((img_name, img_id))

    def _load_saved_files(self) -> None:
        self._load_id2int()
        self._load_int2name()

    def _load_id2int(self, file_n: str = "id2int.json") -> None:
        with open(os.path.join(self._base_dir, file_n), "r", encoding="utf-8") as f:
            self._id2int = json.load(f)

    def _load_int2name(self, file_n: str = "int2name.json") -> None:
        with open(os.path.join(self._base_dir, file_n), "r", encoding="utf-8") as f:
            # convert the keys back to int
            self._int2name = json.load(f, object_hook=lambda item: {int(k): v for k, v in item.items()})

    def transform(image):
        original_size = image.size
        # image.show()
        new_height = random.randint(0, original_size[1])
        new_width = random.randint(0, original_size[0])
        crop_size = [new_height, new_width]
        # print(crop_size)

        transformimage = vision.Compose(
            [
                vision.RandomHorizontalFlip(p=0.5),
                vision.transforms.RandomCrop(crop_size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
                vision.Resize(original_size)
            #  transforms.ToTensor()
                ])
        image = transformimage(image)
        # image.show()

        return image

# Just for testing
if __name__ == '__main__':
    test = TinyImagenetVal(base_dir="../dataset/")
    print("Total dataset size: %s" % len(test))
    print(test[50][0].shape)
    print(test[50][0].type())
    print("Label int: %s\tLabel name: %s" % (test[50][1], test.get_class_name(test[50][1])))
