from pathlib import Path
import os
import random
import json
import itertools
import copy

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, \
    SequentialSampler
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import cv2
import PIL
import scipy.io
import cv2

from . import utils

from PIL import Image

from test_info import CHANNELS, DATASET_PATH

class SevenChannelTrans(object):
    def __init__(self, file_path, patch_size=7):
        self.file_path = file_path
        self.patch_size = patch_size
        self.counter = 0

    def __call__(self, image):
        org_image = torch.clone(image)
        file_path = str(self.file_path)
        for chan in CHANNELS:
            channel_path = file_path.replace("<INSERT_HERE>", chan['dir']).replace("<ENDING>", chan['end'])
            if os.path.isfile(channel_path):
                img = Image.open(channel_path)
                img = transforms.ToTensor()(np.array(img))
                print(image.shape)
                if len(image.size()) == 2:
                    img = torch.unsqueeze(img, 0)
                image = torch.cat((image, img), 0)
            else:
                method = getattr(self, chan['func'])
                new_channel = method(org_image)
                save_image(new_channel, channel_path)
                image = torch.cat((image, new_channel), 0)

        '''
        rgb_filepath = file_path[:-16] + "RGB_" + file_path[-16:]
        dark_filepath = file_path[:-16] + "DARK_" + file_path[-16:]

        if os.path.isfile(rgb_filepath):
            rgb_image = Image.open(rgb_filepath)
            rgb_image = transforms.ToTensor()(np.array(rgb_image))
            image = torch.cat((image, rgb_image), 0)
        else:
            mean_layers = self.make_rgb_mean_layer(image)
            save_image(mean_layers, rgb_filepath)
            image = torch.cat((image, mean_layers), 0)

        if os.path.isfile(dark_filepath):
            dark_image = Image.open(dark_filepath).convert('L')
            dark_image = transforms.ToTensor()(np.array(dark_image))
            image = torch.cat((image, dark_image), 0)
        else:   
            dark_layers = self.make_dark_channel(image)
            save_image(dark_layers, dark_filepath)
            image = torch.cat((image, dark_layers), 0)
        
        self.counter += 1
        '''

        return image

    def make_rgb_mean_layer(self, img):
        image = torch.clone(img)
        means = image.mean((1,2))
        for i in range(3):
            image[i] -= means[i]
        image[image < 0] = 0
        return image

    #patch size if the distance in either direction that the algorithm will look in
    def make_dark_channel(self, img, patch_size = 7):
        image = torch.clone(img)
        height = image.shape[1]
        width = image.shape[2]
        dark = torch.zeros((height, width))
        for i in range(height):
            for j in range(width):
                patch = image[:, max(0, i - patch_size):min(i+patch_size + 1, height), max(0, j - patch_size):min(j + patch_size + 1, width)]
                dark[i][j] = torch.min(patch)
        return dark.unsqueeze(0)
        



