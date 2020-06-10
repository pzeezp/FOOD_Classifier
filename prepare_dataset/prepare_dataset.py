from __future__ import print_function

import re
import json
import torch.utils.data as data


from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomHorizontalFlip, RandomRotation
import torch
import torch.nn.functional as F


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, dir_json, dir_target, num_classes, input_transform=None):
        super(DatasetFromFolder, self).__init__()

        self.data = []
        self.target = []

        with open(dir_json, "r") as read_file:
            data_file = json.load(read_file)

        with open(dir_target, "r") as f:
            line = f.readline()
            name_classes = []
            number_labels = []
            count = 0
            while line:
                line = re.sub(r"[\n]", "", line)
                name_classes.append(line)
                number_labels.append(count)
                line = f.readline()
                count += 1
        label_to_torch = torch.from_numpy(np.array(number_labels))
        labels = F.one_hot(label_to_torch, num_classes=num_classes)

        for x in range(0, len(name_classes)):
            data_one_class = []
            label_name = name_classes[x]
            image_name = data_file[label_name]
            data_one_class.append(image_name)
            mas = np.array(data_one_class).flatten()
            reduceLen_mas = len(mas)
            for ob in range(0, int(reduceLen_mas)):
                self.data.append(mas[ob])
                self.target.append(labels[x])

        self.input_transform = input_transform

    def __getitem__(self, index):

        input = load_img("FooD_Dataset/images/" + self.data[index] + ".jpg")
        target = self.target[index]

        if self.input_transform:
            input = self.input_transform(input)

        return input, target.type(torch.FloatTensor)

    def __len__(self):
        return len(self.data)


def input_transform():
    return Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(),
        RandomRotation(degrees=15),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def test_transform():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])


def get_training_set(num_classes):
    train_dir = "FooD_Dataset/meta/meta/train.json"
    train_label_dir = "FooD_Dataset/meta/meta/classes.txt"

    return DatasetFromFolder(train_dir, train_label_dir, num_classes,
                             input_transform=input_transform())


def get_test_set(num_classes):
    test_dir = "FooD_Dataset/meta/meta/test.json"
    test_label_dir = "FooD_Dataset/meta/meta/classes.txt"

    return DatasetFromFolder(test_dir, test_label_dir, num_classes,
                             input_transform=test_transform())
