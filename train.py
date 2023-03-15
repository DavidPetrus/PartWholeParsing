import numpy as np
import cv2
import torch
import torch.nn.functional as F
import glob
import datetime
import clip
import time
import copy
import pickle as pkl

from dataloader import ADE20k_2017, ADE_Challenge, Coco
from model import ImageParser
import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('data_dir', '/mnt/lustre/users/dvanniekerk1', '')
flags.DEFINE_string('dataset','coco','ADE_Partial, ADE_Full, coco')
flags.DEFINE_integer('batch_size',64,'')
flags.DEFINE_float('lr',0.0003,'')
flags.DEFINE_integer('num_workers',8,'')
flags.DEFINE_integer('image_size',224,'')
flags.DEFINE_integer('num_crops',2,'')
flags.DEFINE_float('min_crop',0.55,'Height/width size of crop')
flags.DEFINE_float('max_crop',0.95,'Height/width size of crop')

flags.DEFINE_integer('num_prototypes',100,'')
flags.DEFINE_integer('depth',3,'')
flags.DEFINE_integer('topk_reg', 10, '')
flags.DEFINE_float('entropy_coeff',0.1,'')
flags.DEFINE_float('entropy_temp',0.1,'')
flags.DEFINE_integer('embd_dim',384,'')
flags.DEFINE_integer('output_dim',70,'')

flags.DEFINE_float('aug_strength',0.7,'')


def main(argv):
    wandb.init(project="PartWholeParsing",name=FLAGS.exp)
    wandb.config.update(flags.FLAGS)

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

    elif FLAGS.dataset == 'coco':
        training_set = Coco("train")
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

        validation_set = Coco("val")
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)


    model = ImageParser("vit_small")

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

    train_iter = 0
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(100):
        for data in training_generator:
            optimizer.zero_grad()

            image_crops, labels, crop_dims = data
            features_a = model(image_crops[0].to('cuda'))[:,1:].reshape(FLAGS.batch_size, FLAGS.image_size//8, FLAGS.image_size//8, FLAGS.output_dim)
            features_b = model(image_crops[1].to('cuda'))[:,1:].reshape(FLAGS.batch_size, FLAGS.image_size//8, FLAGS.image_size//8, FLAGS.output_dim)

            dists_a = ((features_a.unsqueeze(-2) - model.prototypes)**2).mean(dim=-1).reshape(FLAGS.batch_size, FLAGS.image_size//8, FLAGS.image_size//8, FLAGS.num_prototypes)
            dists_b = ((features_b.unsqueeze(-2) - model.prototypes)**2).mean(dim=-1).reshape(FLAGS.batch_size, FLAGS.image_size//8, FLAGS.image_size//8, FLAGS.num_prototypes)

            feats_crop_a, feats_crop_b = model.match_crops(features_a, features_b, crop_dims)
            dists_crop_a, dists_crop_b = model.match_crops(dists_a, dists_b, crop_dims)

            protos_a = model.prototypes[torch.argmin(dists_crop_a, dim=-1)]
            protos_b = model.prototypes[torch.argmin(dists_crop_b, dim=-1)]

            loss_a = ((feats_crop_a - protos_b)**2).mean(dim=-1).mean()
            loss_b = ((feats_crop_b - protos_a)**2).mean(dim=-1).mean()

            top_k_dists_a = torch.topk(-dists_a, FLAGS.topk_reg, -1)[0]
            top_k_dists_b = torch.topk(-dists_b, FLAGS.topk_reg, -1)[0]
            #print(top_k_dists_a)
            entropy_reg = -torch.log(F.softmax(top_k_dists_a/FLAGS.entropy_temp, dim=-1).mean(dim=(1,2))).mean() \
                          -torch.log(F.softmax(top_k_dists_b/FLAGS.entropy_temp, dim=-1).mean(dim=(1,2))).mean()

            contrastive_loss = loss_a + loss_b + FLAGS.entropy_coeff*entropy_reg

            label_a = labels[0].reshape(-1).to('cuda')
            preds_a = F.interpolate(model.class_pred(F.one_hot(torch.argmin(dists_a, dim=-1), FLAGS.num_prototypes).to(torch.float32)).movedim(3,1), \
                                    size=FLAGS.image_size,mode='nearest').movedim(1,3).reshape(-1, 27)
            pred_loss_a = F.cross_entropy(preds_a, label_a)

            label_b = labels[1].reshape(-1).to('cuda')
            preds_b = F.interpolate(model.class_pred(F.one_hot(torch.argmin(dists_b, dim=-1), FLAGS.num_prototypes).to(torch.float32)).movedim(3,1), \
                                    size=FLAGS.image_size,mode='nearest').movedim(1,3).reshape(-1, 27)
            pred_loss_b = F.cross_entropy(preds_b, label_b)

            loss = pred_loss_a + pred_loss_b + contrastive_loss

            with torch.no_grad():
                acc_a = (preds_a.argmax(dim=-1) == label_a).to(torch.float32).mean()
                acc_b = (preds_b.argmax(dim=-1) == label_b).to(torch.float32).mean()

            loss.backward()
            optimizer.step()

            log_dict = {"Epoch": epoch, "Iter": train_iter, "Total Loss": loss.item(), "Pred Loss A": pred_loss_a.item(), "Pred Loss B": pred_loss_b.item(), \
                        "Acc A": acc_a.item(), "Acc B": acc_b.item(), "Loss A": loss_a.item(), "Loss B": loss_b.item(), "Entropy Reg": entropy_reg.item()}
            if train_iter % 10 == 0:
                print(log_dict)

            train_iter += 1

            wandb.log(log_dict)

        '''val_iter = 0
        val_acc, val_loss, val_sim_dists, val_std_loss, val_cov_loss = 0.,0.,0.,0.,0.
        with torch.no_grad():
            for data in validation_generator:
                

                val_iter += 1
                val_acc += acc
                val_loss += loss
                val_sim_dists += sims
                val_std_loss += std_loss
                val_cov_loss += cov_loss

            log_dict = {"Epoch": epoch, "Val Acc": val_acc/val_iter, "Val Loss": val_loss/val_iter, "Val Sim Dists": val_sim_dists/val_iter, \
                        "Val Std Loss": val_std_loss/val_iter, "Val Cov Loss": val_cov_loss/val_iter}
            print(log_dict)

            wandb.log(log_dict)'''



if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)