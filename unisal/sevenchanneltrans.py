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
    def __call__(self, image):
        mean_layers = self.make_rgb_mean_layer(image)
        dark_channel = self.make_dark_channel(image)
        return image

    def make_rgb_mean_layer(self, img):
        image = torch.clone(img)
        print(image)
        means = image.mean((1,2))
        for i in range(3):
            image[i] -= means[i]
        image[image < 0] = 0
        return image

    def 

        



