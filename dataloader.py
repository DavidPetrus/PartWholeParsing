import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import time
import pickle as pkl
from torchvision.transforms import Resize, InterpolationMode, CenterCrop, Compose, ToTensor, PILToTensor

from utils import random_crop, color_distortion, loadAde20K

from absl import flags

FLAGS = flags.FLAGS


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class ADE20k_2017(torch.utils.data.Dataset):

  def __init__(self, preprocess, index_ade20k, training=True):
        self.preprocess = preprocess
        self.resize = Resize(336//14, interpolation=InterpolationMode.NEAREST)
        self.color_aug = color_distortion(FLAGS.aug_strength*0.8, FLAGS.aug_strength*0.8, \
                                          FLAGS.aug_strength*0.8, FLAGS.aug_strength*0.2)

        self.index_ade20k = index_ade20k

        self.training = training

  def __len__(self):
        return len(self.index_ade20k['filename'])

  def __getitem__(self, index):
        # Select sample
        ix = 0
        while True:
            img_file = '{}/{}/{}'.format('/home/petrus',self.index_ade20k['folder'][index+ix], self.index_ade20k['filename'][index+ix])
            try:
                img_info = loadAde20K(img_file)
                break
            except:
                #print(img_file)
                ix += 1
        class_mask = img_info['class_mask']
        #parts_mask = img_info['partclass_mask'][0]

        img = Image.open(img_file)
        w,h = img.size

        img, crop_dims = random_crop(img)
        img = self.preprocess(img)
        class_mask = torch.from_numpy(class_mask[crop_dims[1]:crop_dims[1]+crop_dims[2],crop_dims[0]:crop_dims[0]+crop_dims[2]])
        #parts_mask = torch.from_numpy(parts_mask[crop_dims[1]:crop_dims[1]+crop_dims[2],crop_dims[0]:crop_dims[0]+crop_dims[2]])

        class_mask = self.resize(class_mask.unsqueeze(0)).squeeze()
        #parts_mask = self.resize(parts_mask)

        return img, class_mask


class ADE_Challenge(torch.utils.data.Dataset):

  def __init__(self, preprocess, image_files, training=True):
        self.preprocess = preprocess
        self.color_aug = color_distortion(FLAGS.aug_strength*0.8, FLAGS.aug_strength*0.8, \
                                          FLAGS.aug_strength*0.8, FLAGS.aug_strength*0.2)

        self.image_files = image_files

        self.training = training
        n_px = 336//14
        self.transform = Compose([
                            Resize(n_px, interpolation=InterpolationMode.NEAREST),
                            CenterCrop(n_px),
                            PILToTensor()
                        ])

  def __len__(self):
        return len(self.image_files)

  def __getitem__(self, index):
        # Select sample
        img = Image.open(self.image_files[index])
        w,h = img.size

        img, crop_dims = random_crop(img)
        img = self.preprocess(img)

        ann = Image.open(self.image_files[index].replace('images','annotations').replace('.jpg','.png')).crop((crop_dims[0],crop_dims[1],crop_dims[0]+crop_dims[2],crop_dims[1]+crop_dims[2]))
        ann = self.transform(ann).long()[0]

        return img, ann
