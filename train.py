import numpy as np
import cv2
import torch
import glob
import datetime
import clip
import time

from dataloader import ADE20k_Dataset
from model import ImageParser
import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_integer('batch_size',32,'')
flags.DEFINE_bool('train_clip',False,'')
flags.DEFINE_float('lr',0.0003,'')
flags.DEFINE_integer('num_workers',4,'')
flags.DEFINE_integer('image_size',512,'')
flags.DEFINE_float('min_crop',0.3,'Height/width size of crop')
flags.DEFINE_float('max_crop',0.5,'Height/width size of crop')
flags.DEFINE_integer('proj_dim',128,'')
flags.DEFINE_integer('proj_depth',3,'')

flags.DEFINE_float('sim_coeff',1.,'')
flags.DEFINE_float('std_coeff',1.,'')
flags.DEFINE_float('cov_coeff',0.1,'')

flags.DEFINE_float('aug_strength',0.7,'')


def main(argv):
    wandb.init(project="PartWholeParsing",name=FLAGS.exp)

    clip_model, preprocess = clip.load("RN50x4", device='cuda')

    training_set = ADE20k_Dataset(preprocess)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    #validation_set = ADE20k_Dataset(preprocess)
    #validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    model = ImageParser(clip_model).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

    train_iter = 0
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(100):
        for data in training_generator:
            images, whole_masks = data

            feature_maps = model.get_feature_maps(images.to('cuda'))
            sim_dists = model.compute_dists(feature_maps, whole_masks.to('cuda')).mean()
            std_loss,cov_loss,_ = model.vic_reg(feature_maps.movedim(1,3).reshape(FLAGS.batch_size, 36*36, FLAGS.proj_dim))

            loss = FLAGS.sim_coeff*sim_dists + FLAGS.std_coeff*std_loss + FLAGS.cov_coeff*cov_loss
            loss.backward()
            optimizer.step()

            log_dict = {"Epoch": epoch, "Iter": train_iter, "Loss": loss, "Sim Dists": sim_dists, "Std Loss": std_loss, "Cov Loss": cov_loss}
            if train_iter % 1 == 0:
                print(log_dict)

            train_iter += 1

            wandb.log(log_dict)



if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)