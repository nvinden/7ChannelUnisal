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

from . import utils

from PIL import Image

class SevenChannelTrans(object):
    def __init__(self, file_path, patch_size=7):
        self.file_path = file_path
        self.patch_size = patch_size

    def __call__(self, image):
        file_path = str(self.file_path)
        rgb_filepath = file_path[:-16] + "RGB_" + file_path[-16:]
        dark_filepath = file_path[:-16] + "DARK_" + file_path[-16:]

        if os.path.isfile(rgb_filepath):
            inp = np.load(self.file_path)
            image = torch.cat((image, inp), 0)
        else:
            mean_layers = self.make_rgb_mean_layer(image)
            dark_channel = self.make_dark_channel(image)
            image = torch.cat((image, mean_layers, dark_channel), 0)
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
        



