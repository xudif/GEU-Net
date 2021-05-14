from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random


def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy


class TrainDataset(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): csv file path
        """
        # read csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # The first column of the file contains the first image
        self.image1_arr = np.asarray(self.data_info.iloc[:, 0])
        # The second column of the file contains the second image
        self.image2_arr = np.asarray(self.data_info.iloc[:, 1])
        # The third column of the file contains the label
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])
        # compute length
        self.data_len = len(self.data_info.index)

    def __len__(self):
        return self.data_len

    def add_gaussian_noise(self, image_in, noise_sigma):
        image_in = np.array(image_in)
        temp_image = np.float64(np.copy(image_in))

        h = temp_image.shape[0]
        w = temp_image.shape[1]
        noise = np.random.randn(h, w) * noise_sigma

        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
            noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
            noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
        """
        print('min,max = ', np.min(noisy_image), np.max(noisy_image))
        print('type = ', type(noisy_image[0][0][0]))
        """
        noisy_image = Image.fromarray(np.uint8(noisy_image))
        return noisy_image

    def transform(self, img1, img2, label):
        angle = transforms.RandomRotation.get_params([-180, 180])
        # rotate
        img1 = tf.rotate(img1, angle, resample=Image.NEAREST)
        img2 = tf.rotate(img2, angle, resample=Image.NEAREST)
        label = tf.rotate(label, angle, resample=Image.NEAREST)
        # flip
        if random.random() > 0.5:
            img1 = tf.hflip(img1)
            img2 = tf.hflip(img2)
            label = tf.hflip(label)
        if random.random() > 0.5:
            img1 = tf.vflip(img1)
            img2 = tf.vflip(img2)
            label = tf.vflip(label)
        # resize
        H, W = 256, 256
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img1, scale=(0.25, 1.0), ratio=(1, 1))
            img1 = tf.resized_crop(img1, i, j, h, w, (H, W))
            img2 = tf.resized_crop(img2, i, j, h, w, (H, W))
            label = tf.resized_crop(label, i, j, h, w, (H, W))
        else:
            pad = random.randint(0, 192)
            img1 = tf.pad(img1, pad)
            img1 = tf.resize(img1, (H, W))
            img2 = tf.pad(img2, pad)
            img2 = tf.resize(img2, (H, W))
            label = tf.pad(label, pad)
            label = tf.resize(label, (H, W))
        # add gaussian noise
        parm_noise = np.random.uniform()
        img1 = self.add_gaussian_noise(img1, parm_noise)
        img2 = self.add_gaussian_noise(img2, parm_noise)
        # convert to tensor and normalize
        img1 = tf.to_tensor(img1)
        img1 = tf.normalize(img1, [0.44], [0.26])
        img2 = tf.to_tensor(img2)
        img2 = tf.normalize(img2, [0.44], [0.26])
        label = tf.to_tensor(label)
        return img1, img2, label

    def __getitem__(self, index):
        # get file name
        image1_name = self.image1_arr[index]
        image2_name = self.image2_arr[index]
        image_label = self.label_arr[index]
        # read image file
        img_as_img1 = Image.open(image1_name).convert('L')
        img_as_img2 = Image.open(image2_name).convert('L')
        # read image label
        img_label = Image.open(image_label).convert('L')
        # transform
        img1, img2, img_label = self.transform(img_as_img1, img_as_img2, img_label)
        img_train = torch.cat((img1, img2), dim=0)
        # # test
        # self.imshow(img_train[0].squeeze(),title='img1')
        # self.imshow(img_train[1].squeeze(),title='img2')
        # self.imshow(img_label.squeeze(),title='label')
        # print('img_train[0]',img_train[0].shape)
        # print('img_train[1]',img_train[1].shape)
        # print('img_label',img_label.shape)
        # np.savetxt('img1.csv', img_train[0].squeeze(), delimiter=',')
        # np.savetxt('img2.csv', img_train[1].squeeze(), delimiter=',')
        # np.savetxt('label.csv', img_label.squeeze(), delimiter=',')
        # img1 = pd.DataFrame(list(img_train[0].squeeze()))
        # img1.to_csv('img1.txt')
        # img2 = pd.DataFrame(list(img_train[1].squeeze()))
        # img2.to_csv('img2.txt')
        # label = pd.DataFrame(list(img_label.squeeze()))
        # label.to_csv('label.txt')
        return img_train, img_label

    def imshow(self, tensor, title=None):
        image = tensor.clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        unloader = transforms.ToPILImage()
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(2)  # pause a bit so that plots are updated


class Valid_Dataset(Dataset):
    def __init__(self, transforms=None, path_val=''):
        img1_path = path_val + '/image1'
        img2_path = path_val + '/image2'
        root_path1 = os.path.abspath(img1_path)
        root_path2 = os.path.abspath(img2_path)
        self.img1_list = [os.path.join(root_path1, x) for x in os.listdir(img1_path) if x.endswith('.png')]
        self.img2_list = [os.path.join(root_path2, x) for x in os.listdir(img2_path) if x.endswith('.png')]
        self.transforms = transforms

    def __getitem__(self, index):
        # stuff
        image1 = Image.open(self.img1_list[index]).convert('L')
        image2 = Image.open(self.img2_list[index]).convert('L')
        if self.transforms is not None:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
        img_test = torch.cat((image1, image2), dim=0)
        return img_test

    def __len__(self):
        return len(self.img1_list)


class Test_Dataset(Dataset):
    def __init__(self, transforms=None, path_test=''):
        img1_path = path_test + '/image1'
        img2_path = path_test + '/image2'
        root_path1 = os.path.abspath(img1_path)
        root_path2 = os.path.abspath(img2_path)
        self.img1_list = [os.path.join(root_path1, x) for x in os.listdir(img1_path) if x.endswith('.png')]
        self.img2_list = [os.path.join(root_path2, x) for x in os.listdir(img2_path) if x.endswith('.png')]
        self.transforms = transforms

    def __getitem__(self, index):
        # stuff
        image1 = Image.open(self.img1_list[index]).convert('L')
        image2 = Image.open(self.img2_list[index]).convert('L')
        if self.transforms is not None:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
        img_test = torch.cat((image1, image2), dim=0)
        return img_test

    def __len__(self):
        return len(self.img1_list)
