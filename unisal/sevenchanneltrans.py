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

class SevenChannelTrans(object):
    def __init__(self, file_path, patch_size=7):
        self.file_path = file_path
        self.patch_size = patch_size

    def __call__(self, image):
        file_path = str(self.file_path)
        rgb_filepath = file_path[:-16] + "RGB_" + file_path[-16:]
        dark_filepath = file_path[:-16] + "DARK_" + file_path[-16:]

        if os.path.isfile(rgb_filepath):
            print("Found RGB")
        else:
            mean_layers = self.make_rgb_mean_layer(image)
            save_image(mean_layers, rgb_filepath)
            image = torch.cat((image, mean_layers), 0)
            print("Saved RGB: " + rgb_filepath)

        if os.path.isfile(dark_filepath):
            print("Found BLACK")
        else:
            dark_layers = self.make_dark_channel(image)
            save_image(dark_layers, dark_filepath)
            image = torch.cat((image, dark_layers), 0)
            print("Saved BLACK: " + dark_filepath)
        
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
        


