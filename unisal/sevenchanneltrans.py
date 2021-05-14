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
from matplotlib import cm

from . import utils

from PIL import Image, ImageOps

from test_info import CHANNELS, DATASET_PATH

from unisal.models.AdaBins.infer import InferenceHelper

if any(d['dir'] == 'depth_kitti' for d in CHANNELS):
    kitti_helper = InferenceHelper(dataset='kitti', device="cpu")
if any(d['dir'] == 'depth_nyu' for d in CHANNELS):
    nyu_helper = InferenceHelper(dataset='nyu', device="cpu")

class SevenChannelTrans(object):
    def __init__(self, file_path, patch_size=7):
        self.file_path = file_path
        self.patch_size = patch_size
        self.counter = 0

        if any(d['dir'] == 'depth_kitti' for d in CHANNELS):
            self.nyu_helper = InferenceHelper(dataset='nyu', device="cpu")
        if any(d['dir'] == 'depth_nyu' for d in CHANNELS):
            self.kitti_helper = InferenceHelper(dataset='kitti', device="cpu")


    def __call__(self, image):
        org_image = torch.clone(image)
        height = image.shape[1]
        width = image.shape[2]
        file_path = str(self.file_path)
        for chan in CHANNELS:
            channel_path = file_path.replace("<INSERT_HERE>", chan['dir']).replace("<ENDING>", chan['end'])
            if os.path.isfile(channel_path):
                img = Image.open(channel_path)
                if chan['chan'] == 1:
                    img = img.convert("L")
                img = transforms.ToTensor()(np.array(img))
                #if not the same size as image
                if img.shape[1] != height or img.shape[2] != width:
                    img = transforms.Resize((height, width))(img)
                    save_image(img, channel_path)
                #print(f"{chan['dir']}:{img.shape}:{chan['chan']}")
                image = torch.cat((image, img), 0)
            else:
                print("method")
                method = getattr(self, chan['func'])
                new_channel = method(org_image)
                if new_channel.shape[1] != height or new_channel.shape[2] != width:
                    new_channel = transforms.Resize((height, width))(new_channel)
                save_image(new_channel, channel_path)
                image = torch.cat((image, new_channel), 0)

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
    
    def depth_kitti(self, img):
        img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
        im = Image.fromarray(img, "RGB")
        im.save("TEST_KITTI.jpg")
        _, predicted_depth = self.kitti_helper.predict_pil(im)
        return predicted_depth

    def depth_nyu(self, img):
        img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
        im = Image.fromarray(img, "RGB")
        _, predicted_depth = self.nyu_helper.predict_pil(im)
        return predicted_depth

    

    
        



