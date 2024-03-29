from __future__ import division
import os
import numpy as np
import glob
import torch
import cv2
from torch.utils.data import Dataset

import torchvision.datasets as dset
import torchvision.transforms as transforms

from config import DATA_PATH, IMAGENET_PATH, GTSRB_PATH
from dataset_utils.gtsrb_utils import preprocess_gtsrb_img
from utils.dataset import get_class_specific_dataset_folder_name, generate_separated_dataset


class DatasetFromNumpy(Dataset):
    def __init__(self, images, labels,
                 transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])):
        self.images = images
        self.labels = torch.tensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        if self.transform is not None:
            X = self.transform(X)

        y = self.labels[index]
        return X, y


def get_data_specs(dataset, architecture):
    mean = [0., 0., 0.]
    std = [1., 1., 1.]

    if dataset == 'cifar10':
        num_classes = 10
        input_size = 32
        num_channels = 3

    elif dataset == 'cifar100':
        num_classes = 100
        input_size = 32
        num_channels = 3

    elif dataset == "gtsrb":
        num_classes = 43
        input_size = 48
        num_channels = 3

    elif dataset == "imagenet":
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224
        num_channels = 3
    else:
        raise ValueError()
    return num_classes, (mean, std), input_size, num_channels


def get_data(dataset, mean, std, input_size, classes=None, others=False, train_target_model=False,
             train_samples_per_class=-1):
    if train_target_model:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(input_size, padding=4),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    if dataset == 'cifar10':
        train_data = dset.CIFAR10(DATA_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(DATA_PATH, train=False, transform=test_transform, download=True)
        if train_samples_per_class > 0:
            data = np.zeros((0, 32, 32, 3), dtype='uint8')
            targets = np.array([], dtype=int)
            for cl in range(10):
                idxs = np.array(train_data.targets) == cl
                data = np.concatenate((data, train_data.data[idxs][0:train_samples_per_class]))
                targets = np.concatenate((targets, np.array(train_data.targets)[idxs][0:train_samples_per_class]))
            train_data.data = data
            train_data.targets = targets.tolist()
    elif dataset == 'cifar100':
        train_data = dset.CIFAR100(DATA_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(DATA_PATH, train=False, transform=test_transform, download=True)
    elif dataset == "gtsrb":
        gtsrb_train_path = os.path.join(GTSRB_PATH, "Training")
        gtsrb_test_path = os.path.join(GTSRB_PATH, "Testing")

        gtsrb_num_train_imgs = 26640
        gtsrb_num_test_imgs = 12630

        # Numpy Training data
        train_img_paths = glob.glob(os.path.join(gtsrb_train_path, '*/*.ppm'))
        assert len(train_img_paths) == gtsrb_num_train_imgs
        X_train = np.zeros((gtsrb_num_train_imgs, input_size, input_size, 3), dtype=np.uint8)
        Y_train = []

        for idx, img_path in enumerate(train_img_paths):
            input_img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            X_train[idx] = preprocess_gtsrb_img(input_img, img_size=input_size)

            label = int(img_path.split('/')[-2])
            Y_train.append(label)

        # Numpy Testing data
        test_img_paths = glob.glob(os.path.join(gtsrb_test_path, '*/*.ppm'))
        assert len(test_img_paths) == gtsrb_num_test_imgs
        X_test = np.zeros((gtsrb_num_test_imgs, input_size, input_size, 3), dtype=np.uint8)
        Y_test = []

        for idx, img_path in enumerate(test_img_paths):
            input_img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            X_test[idx] = preprocess_gtsrb_img(input_img, img_size=input_size)
            label = int(img_path.split('/')[-2])
            Y_test.append(label)

        if train_target_model:
            train_transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(input_size, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            train_transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])

        test_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])

        if classes is not None:
            train_idxs = [lbl in classes for lbl in Y_train]
            X_train = X_train[train_idxs]
            Y_train = np.array(Y_train)[train_idxs].tolist()

            test_idxs = [lbl in classes for lbl in Y_test]
            X_test = X_test[test_idxs]
            Y_test = np.array(Y_test)[test_idxs].tolist()

        train_data = DatasetFromNumpy(X_train, Y_train, transform=train_transform)
        test_data = DatasetFromNumpy(X_test, Y_test, transform=test_transform)

    elif dataset == "imagenet":
        traindir = os.path.join(IMAGENET_PATH, 'train')
        valdir = os.path.join(IMAGENET_PATH, 'val')

        if train_target_model:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        if input_size == 299:
            train_transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

            test_transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        if classes is not None:
            separated_imagenet_path = get_class_specific_dataset_folder_name(dataset, classes,
                                                                             samples_per_class=train_samples_per_class)
            if (not os.path.exists(separated_imagenet_path)) & (not others):
                generate_separated_dataset(dataset, classes, samples_per_class=train_samples_per_class)
            if others == False:
                traindir = os.path.join(separated_imagenet_path, 'source_classes_train')
                valdir = os.path.join(separated_imagenet_path, 'source_classes_val')
            elif others == True:
                traindir = os.path.join(separated_imagenet_path, 'other_classes_train')
                valdir = os.path.join(separated_imagenet_path, 'others_classes_val')
            else:
                raise ValueError()

        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data = dset.ImageFolder(root=valdir, transform=test_transform)

    if classes is not None:
        if dataset in ['cifar10', 'cifar100']:
            # cifar10 train:50000, test:10000
            train_idxs = [lbl in classes for lbl in train_data.targets]

            targets = train_data.targets
            class_num = 5000
            for class_s in classes:
                count = 0
                i = 0
                for lbl in targets:
                    if class_s == lbl and count <= class_num:
                        count += 1
                    if class_s == lbl and count > class_num:
                        train_idxs[i] = False
                    i += 1

            train_data.data = train_data.data[train_idxs]
            train_data.targets = np.array(train_data.targets)[train_idxs].tolist()

            test_idxs = [lbl in classes for lbl in test_data.targets]
            test_data.data = test_data.data[test_idxs]
            test_data.targets = np.array(test_data.targets)[test_idxs].tolist()

    return train_data, test_data
