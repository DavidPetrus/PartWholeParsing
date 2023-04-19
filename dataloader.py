import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import time
import pickle as pkl
import glob
import cv2
import random
import datetime
from torchvision.transforms import Resize, CenterCrop, Compose, ToTensor, PILToTensor, ColorJitter

from utils import random_crop, color_distortion, loadAde20K, color_normalize, transform_image

from absl import flags

FLAGS = flags.FLAGS


def _convert_image_to_rgb(image):
    return image.convert("RGB")

class Cityscapes(torch.utils.data.Dataset):
    def __init__(self, image_set):
        super(Cityscapes, self).__init__()

        if image_set == 'train':
            self.image_files = glob.glob(f"{FLAGS.data_dir}/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/*/*")
            #                   glob.glob(f"{FLAGS.data_dir}/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/*/*")
        else:
            self.image_files = glob.glob(f"{FLAGS.data_dir}/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/*/*")

        self.color_jitter = ColorJitter(brightness=FLAGS.aug_strength, contrast=FLAGS.aug_strength, saturation=FLAGS.aug_strength, hue=0.2*FLAGS.aug_strength)


    def __getitem__(self, index):

        img_batch = [[] for c in range(FLAGS.num_crops)]
        label_batch = [[] for c in range(FLAGS.num_crops)]
        crop_dims = []
        for c in range(FLAGS.num_crops):
            crop_size = np.random.uniform(FLAGS.min_crop, 1.)
            crop_dims.append([np.random.uniform(0.,1.-crop_size), np.random.uniform(0.,1.-crop_size), crop_size])

        batch_sample = random.sample(self.image_files, FLAGS.batch_size)

        for img_file in batch_sample:
            img = cv2.imread(img_file)
            label_file = img_file.replace("leftImg8bit_trainvaltest/leftImg8bit", "gtFine_trainvaltest/gtFine").replace("leftImg8bit", "gtFine_labelIds")
            label = torch.as_tensor(np.array(Image.open(label_file)), dtype=torch.float32)
            label -= 7
            label[label < 0] = -1

            img = img[:,:,::-1]
            img_h, img_w, _ = img.shape

            # Make image square
            if img_w > img_h:
                square_x = np.random.randint(0, img_w-img_h)
                img = img[:, square_x: square_x+img_h]
                label = label[:, square_x: square_x+img_h]
            elif img_h > img_w:
                square_y = np.random.randint(0, img_h-img_w)
                img = img[square_y: square_y+img_w, :]
                label = label[:, square_x: square_x+img_h]

            img = transform_image(img)

            for c in range(FLAGS.num_crops):
                cr = random_crop(img, crop_dims=crop_dims[c])
                lab_crop = random_crop(label.unsqueeze(0), crop_dims=crop_dims[c], inter_mode='nearest')
                img_batch[c].append(color_normalize(self.color_jitter(cr)))
                label_batch[c].append(lab_crop)

                if lab_crop.max() < 0:
                    print(img_file)

        return [torch.cat(cr,dim=0) for cr in img_batch], [torch.cat(cr,dim=0) for cr in label_batch], crop_dims


    def __len__(self):
        return len(self.image_files) // FLAGS.batch_size


class CelebA(torch.utils.data.Dataset):
    def __init__(self, image_set):
        super(CelebA, self).__init__()

        self.image_files = glob.glob(f"{FLAGS.data_dir}/CelebAMask-HQ/CelebA-HQ-img/*")
        if image_set == 'train':
            self.image_files = self.image_files[:-3000]
        else:
            self.image_files = self.image_files[-3000:]

        self.color_jitter = ColorJitter(brightness=FLAGS.aug_strength, contrast=FLAGS.aug_strength, saturation=FLAGS.aug_strength, hue=0.2*FLAGS.aug_strength)


    def __getitem__(self, index):

        img_batch = [[] for c in range(FLAGS.num_crops)]
        label_batch = [[] for c in range(FLAGS.num_crops)]
        crop_dims = []
        for c in range(FLAGS.num_crops):
            crop_size = np.random.uniform(FLAGS.min_crop,1.)
            crop_dims.append([np.random.uniform(0.,1.-crop_size),np.random.uniform(0.,1.-crop_size),crop_size])

        batch_sample = random.sample(self.image_files, FLAGS.batch_size)

        for img_file in batch_sample:
            img_idx = int(img_file.split('/')[-1][:-4])
            '''label_files = glob.glob(img_file.replace('CelebA-HQ-img', f'CelebAMask-HQ-mask-anno/{int(img_ix//2000)}/{img_ix:05d}_*'))
            labels = {}
            for lab_file in label_files:
                cat = lab_file.split('_')[-1][:-4]
                labels[cat] = torch.as_tensor(np.array(Image.open(lab_file)), dtype=torch.int64)'''

            img = cv2.imread(img_file)

            img = img[:,:,::-1]
            h,w,_ = img.shape

            img = transform_image(img)

            for c in range(FLAGS.num_crops):
                cr = random_crop(img, crop_dims=crop_dims[c])
                #lab_crop = random_crop(coarse_label.unsqueeze(0).to(torch.float32), crop_dims=crop_dims[c], inter_mode='nearest').to(torch.long)
                img_batch[c].append(color_normalize(self.color_jitter(cr)))
                #label_batch[c].append(lab_crop)

        return [torch.cat(cr,dim=0) for cr in img_batch], [None] * FLAGS.batch_size, crop_dims


    def __len__(self):
        return len(self.image_files) // FLAGS.batch_size


class Coco(torch.utils.data.Dataset):
    def __init__(self, image_set):
        super(Coco, self).__init__()
        self.split = image_set

        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }

        self.image_files = []
        self.label_files = []

        for split_dir in split_dirs[self.split]:
            self.image_files = glob.glob(f"{FLAGS.data_dir}/coco/{split_dir}/*.jpg")
            self.label_files = glob.glob(f"{FLAGS.data_dir}/coco/stuffthingmaps_trainval2017/{split_dir}/*.png")
            self.image_files.sort()
            self.label_files.sort()

        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

        self.color_jitter = ColorJitter(brightness=FLAGS.aug_strength, contrast=FLAGS.aug_strength, saturation=FLAGS.aug_strength, hue=0.2*FLAGS.aug_strength)


    def __getitem__(self, index):

        img_batch = [[] for c in range(FLAGS.num_crops)]
        label_batch = [[] for c in range(FLAGS.num_crops)]
        crop_dims = []
        for c in range(FLAGS.num_crops):
            crop_size = np.random.uniform(FLAGS.min_crop,1.)
            crop_dims.append([np.random.uniform(0.,1.-crop_size),np.random.uniform(0.,1.-crop_size),crop_size])

        batch_sample = random.sample(list(zip(self.image_files,self.label_files)),FLAGS.batch_size)

        for img_file, label_file in batch_sample:
            while True:
                try:
                    img = cv2.imread(img_file)
                    label = torch.as_tensor(np.array(Image.open(label_file)), dtype=torch.int64)
                    if img is None: 
                        raise
                    break
                except Exception as e:
                    print(e)
                    print('-----------------------', img_file)
                    idx = np.random.randint(len(self.image_files))
                    img_file = self.image_files[idx]
                    label_file = self.label_files[idx]

            label[label == 255] = -1

            if FLAGS.num_output_classes == 28:
                coarse_label = torch.zeros_like(label)
                for fine, coarse in self.fine_to_coarse.items():
                    coarse_label[label == fine] = coarse
            else:
                coarse_label = label

            coarse_label[coarse_label == -1] = FLAGS.num_output_classes-1

            img = img[:,:,::-1]
            h,w,_ = img.shape

            img = transform_image(img)

            for c in range(FLAGS.num_crops):
                cr = random_crop(img, crop_dims=crop_dims[c])
                lab_crop = random_crop(coarse_label.unsqueeze(0).to(torch.float32), crop_dims=crop_dims[c], inter_mode='nearest').to(torch.long)
                img_batch[c].append(color_normalize(self.color_jitter(cr)))
                label_batch[c].append(lab_crop)

        return [torch.cat(cr,dim=0) for cr in img_batch], [torch.cat(cr,dim=0).squeeze() for cr in label_batch], crop_dims


    def __len__(self):
        return len(self.image_files) // FLAGS.batch_size


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

        img = random_crop(img)
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

        img = random_crop(img)
        img = self.preprocess(img)

        ann = Image.open(self.image_files[index].replace('images','annotations').replace('.jpg','.png')).crop((crop_dims[0],crop_dims[1],crop_dims[0]+crop_dims[2],crop_dims[1]+crop_dims[2]))
        ann = self.transform(ann).long()[0]

        return img, ann
