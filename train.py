import numpy as np
import cv2
import torch
import glob
import datetime
import clip
import time
import copy
import pickle as pkl

from dataloader import ADE20k_2017, ADE_Challenge
from model import ImageParser
import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('dataset','ADE_Partial','ADE_Partial, ADE_Full')
flags.DEFINE_integer('batch_size',16,'')
flags.DEFINE_bool('train_clip',False,'')
flags.DEFINE_float('lr',0.00003,'')
flags.DEFINE_integer('num_workers',4,'')
flags.DEFINE_integer('image_size',512,'')
flags.DEFINE_float('min_crop',0.4,'Height/width size of crop')
flags.DEFINE_float('max_crop',0.7,'Height/width size of crop')
flags.DEFINE_string('clip_model','vit','vit, resx4, resx16')
flags.DEFINE_integer('proj_dim',768,'')
flags.DEFINE_integer('proj_depth',3,'')

flags.DEFINE_float('sim_coeff',1.,'')
flags.DEFINE_float('std_coeff',10.,'')
flags.DEFINE_float('cov_coeff',100.,'')

flags.DEFINE_float('aug_strength',0.7,'')


def main(argv):
    wandb.init(project="PartWholeParsing",name=FLAGS.exp)
    wandb.config.update(flags.FLAGS)

    if FLAGS.clip_model == 'vit':
        clip_model, preprocess = clip.load("ViT-L/14@336px", device='cuda')
    else:
        clip_model, preprocess = clip.load("RN50x4", device='cuda')

    if FLAGS.dataset == 'ADE_Full':
        with open('/home/petrus/ADE20K_2021_17_01/index_ade20k.pkl', 'rb') as f:
            index_ade20k = pkl.load(f)

        train_index = copy.copy(index_ade20k)
        train_index['filename'] = index_ade20k['filename'][:25258]
        train_index['folder'] = index_ade20k['folder'][:25258]
        train_index['scene'] = index_ade20k['scene'][:25258]

        val_index = copy.copy(index_ade20k)
        val_index['filename'] = index_ade20k['filename'][25258:27257]
        val_index['folder'] = index_ade20k['folder'][25258:27257]
        val_index['scene'] = index_ade20k['scene'][25258:27257]

        training_set = ADE20k_2017(preprocess, train_index, training=True)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)

        validation_set = ADE20k_2017(preprocess, val_index, training=False)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)

        model = ImageParser(clip_model, index_ade20k['objectnames']).to('cuda')

    elif FLAGS.dataset == 'ADE_Partial':
        train_images = glob.glob("/home/petrus/ADE20K/ADEChallengeData2016/images/training/*")
        training_set = ADE_Challenge(preprocess, train_images, training=True)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)

        val_images = glob.glob("/home/petrus/ADE20K/ADEChallengeData2016/images/validation/*")
        validation_set = ADE_Challenge(preprocess, val_images, training=False)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)

        with open("/home/petrus/ADE20K/ADEChallengeData2016/objectInfo150.txt",'r') as fp:
            lines = fp.read().splitlines()

        objects = []
        for line in lines[1:]:
            objects.append(line.split('\t')[-1])

        model = ImageParser(clip_model, objects).to('cuda')

    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.lr)

    train_iter = 0
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(100):
        for data in training_generator:
            optimizer.zero_grad()

            images, whole_masks = data

            feature_maps = model.get_feature_maps(images.to('cuda').half())
            sims = model.compute_sim_loss(feature_maps, whole_masks.to('cuda'))
            std_loss,cov_loss,_ = model.vic_reg(feature_maps.reshape(FLAGS.batch_size, 24*24, FLAGS.proj_dim))

            loss = FLAGS.sim_coeff*sims + FLAGS.std_coeff*std_loss + FLAGS.cov_coeff*cov_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = model.get_acc(feature_maps, whole_masks.to('cuda'))

            log_dict = {"Epoch": epoch, "Iter": train_iter, "Acc": acc, "Loss": loss, "Sim Dists": sims, "Std Loss": std_loss, "Cov Loss": cov_loss}
            if train_iter % 100 == 0:
                print(log_dict)

            train_iter += 1

            wandb.log(log_dict)

        val_iter = 0
        val_acc, val_loss, val_sim_dists, val_std_loss, val_cov_loss = 0.,0.,0.,0.,0.
        with torch.no_grad():
            for data in validation_generator:
                images, whole_masks = data

                feature_maps = model.get_feature_maps(images.to('cuda'))
                sims = model.compute_sim_loss(feature_maps, whole_masks.to('cuda'))
                std_loss,cov_loss,_ = model.vic_reg(feature_maps.reshape(FLAGS.batch_size, 24*24, FLAGS.proj_dim))

                loss = FLAGS.sim_coeff*sims + FLAGS.std_coeff*std_loss + FLAGS.cov_coeff*cov_loss
                acc = model.get_acc(feature_maps, whole_masks.to('cuda'))

                val_iter += 1
                val_acc += acc
                val_loss += loss
                val_sim_dists += sims
                val_std_loss += std_loss
                val_cov_loss += cov_loss

            log_dict = {"Epoch": epoch, "Val Acc": val_acc/val_iter, "Val Loss": val_loss/val_iter, "Val Sim Dists": val_sim_dists/val_iter, \
                        "Val Std Loss": val_std_loss/val_iter, "Val Cov Loss": val_cov_loss/val_iter}
            print(log_dict)

            wandb.log(log_dict)



if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)