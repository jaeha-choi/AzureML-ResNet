import json
import os

import PIL.Image
import numpy as np
import torch.utils.data


class TinyImagenet(torch.utils.data.Dataset):

    def __init__(self, base_dir: str = "../dataset/", src_dir: str = "train", load_all: bool = False, transform=None):
        """
        Initialize the dataset
        :param base_dir: Dataset directory. Contains test/, train/, val/, words.txt, etc
        :param src_dir: Source directory, such as train/, val/, etc.
        :param load_all: If set to True, loads previously saved json files such as id2name.
        """
        # Dataset directory. Contains test/, train/, val/, words.txt, etc
        self.base_dir = base_dir
        self.src_dir = os.path.join(base_dir, src_dir)
        self.word_file_n = os.path.join(base_dir, "words.txt")
        self.images = []  # [("n01443537_0.jpeg", 7335), ...]
        self.id2name = {}  # n01443537 -> "goldfish, Carassius auratus"
        self.id2int = {}  # n01443537 -> 7335
        self.int2name = {}  # 7335 -> "goldfish, Carassius auratus"

        # If some transformation is necessary, create a function for it
        self.transform = transform

        if load_all:
            self.load_all_dict()
        else:
            self.load_words_txt()

        self._load_images()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> (torch.Tensor, int):
        img_name = self.images[idx][0]
        cls_id = self.images[idx][1]
        cls_int = self.id2int[cls_id]

        full_file_name = os.path.join(self.src_dir, cls_id, "images/", img_name)
        # Uncomment the following line for absolute path
        # full_file_name = os.path.abspath(full_file_name)
        image = PIL.Image.open(full_file_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return torch.from_numpy(np.array(image)), cls_int

    def load_words_txt(self) -> None:
        with open(self.word_file_n, "r", encoding="utf-8") as file:
            for line in file:
                ln = str(line).strip().split("\t", maxsplit=2)
                cls_id = ln[0]
                cls_name = ln[1]
                self.id2name[cls_id] = cls_name
                self.id2int[cls_id] = len(self.id2int)
                self.int2name[len(self.int2name)] = cls_name

    def save_id2name(self, file_n: str = "id2name.json") -> None:
        with open(file_n, "w", encoding="utf-8") as file:
            json.dump(self.id2name, file)

    def load_id2name(self, file_n: str = "id2name.json") -> None:
        with open(file_n, "r", encoding="utf-8") as file:
            self.id2name = json.load(file)

    def save_id2int(self, file_n: str = "id2int.json") -> None:
        with open(file_n, "w", encoding="utf-8") as file:
            json.dump(self.id2int, file)

    def load_id2int(self, file_n: str = "id2int.json") -> None:
        with open(file_n, "r", encoding="utf-8") as file:
            self.id2int = json.load(file)

    def save_int2name(self, file_n: str = "int2name.json") -> None:
        with open(file_n, "w", encoding="utf-8") as file:
            json.dump(self.int2name, file)

    def load_int2name(self, file_n: str = "int2name.json") -> None:
        with open(file_n, "r", encoding="utf-8") as file:
            # convert the keys back to int
            self.int2name = json.load(file, object_hook=lambda item: {int(k): v for k, v in item.items()})

    def load_all_dict(self) -> None:
        self.load_id2name()
        self.load_id2int()
        self.load_int2name()

    def save_all_dict(self) -> None:
        self.save_id2name()
        self.save_id2int()
        self.save_int2name()

    def _load_images(self, use_bounding_box: bool = False) -> None:
        if not self.id2int:
            raise KeyError("Load id2int before you load images.")
        if use_bounding_box:
            raise NotImplementedError("Using bounding box is not yet supported.")
        for curr_dir, folder_names, file_names in os.walk(self.src_dir, followlinks=True):
            # If folder_names is empty, this means we're reading a folder that only contains images.
            # Use this option if bounding box data isn't required
            if (not use_bounding_box) and (not folder_names):
                # This gives class id. E.g: n01443537
                class_id = os.path.basename(os.path.dirname(curr_dir))
                for file_name in file_names:
                    # print((file_name, class_id))
                    self.images.append((file_name, class_id))

            # Generate image data by reading class_id_boxes.txt
            # Use this option if bounding box data is required
            elif use_bounding_box and folder_names and folder_names[0] == "images":
                # This gives class id. E.g: n01443537
                class_id = os.path.basename(curr_dir)
                # Reads *class_id*_boxes.txt file that contains bounding box data
                with open(os.path.join(curr_dir, file_names[0])) as file:
                    for line in file:
                        ln = str(line).strip().split("\t")
                        file_name = ln[0]
                        # Bounding box data (left top point, right bot point)
                        point1 = (int(ln[1]), int(ln[2]))
                        point2 = (int(ln[3]), int(ln[4]))
                        # TODO: determine how to store these data and remove NotImplementedError
                        print(file_name, point1, point2, class_id)


# Just for testing
if __name__ == '__main__':
    test = TinyImagenet()
    print(len(test))
    print(test[0])
