import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import time
import pickle as pkl
from torchvision.transforms import Resize, InterpolationMode

from utils import random_crop, color_distortion, loadAde20K

from absl import flags

FLAGS = flags.FLAGS


class ADE20k_Dataset(torch.utils.data.Dataset):

  def __init__(self, preprocess):
        self.preprocess = preprocess
        self.resize = Resize(288//8, interpolation=InterpolationMode.NEAREST)
        self.color_aug = color_distortion(FLAGS.aug_strength*0.8, FLAGS.aug_strength*0.8, \
                                          FLAGS.aug_strength*0.8, FLAGS.aug_strength*0.2)

        with open('/home/petrus/ADE20K_2021_17_01/index_ade20k.pkl', 'rb') as f:
            self.index_ade20k = pkl.load(f)

  def __len__(self):
        return len(self.index_ade20k['filename'])

  def __getitem__(self, index):
        # Select sample
        img_file = '{}/{}/{}'.format('/home/petrus',self.index_ade20k['folder'][index], self.index_ade20k['filename'][index])
        img_info = loadAde20K(img_file)
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
