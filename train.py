import numpy as np
import cv2
import torch
import torch.nn.functional as F
import glob
import datetime
import time
import copy
import pickle as pkl

from dataloader import ADE20k_2017, ADE_Challenge, Coco
from model import ImageParser
from utils import sinkhorn_knopp
import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('data_dir', '/mnt/lustre/users/dvanniekerk1', '')
flags.DEFINE_string('dataset','coco','ADE_Partial, ADE_Full, coco')
flags.DEFINE_bool('feed_labels',False,'')
flags.DEFINE_integer('batch_size',32,'')
flags.DEFINE_float('lr',0.0003,'')
flags.DEFINE_integer('num_workers',8,'')
flags.DEFINE_integer('image_size',224,'')
flags.DEFINE_integer('num_crops',2,'')
flags.DEFINE_float('min_crop',0.55,'Height/width size of crop')
flags.DEFINE_float('max_crop',0.95,'Height/width size of crop')

flags.DEFINE_string('dist_func','cosine','cosine, euc')
flags.DEFINE_integer('num_prototypes',100,'')
flags.DEFINE_float('epsilon',0.05,'')
flags.DEFINE_integer('sinkhorn_iters',3,'')
flags.DEFINE_float('temperature',0.1,'')
flags.DEFINE_bool('round_q',True,'')
flags.DEFINE_integer('depth',3,'')
flags.DEFINE_integer('topk_reg', 10, '')
flags.DEFINE_float('entropy_coeff',0.1,'')
flags.DEFINE_float('entropy_temp',0.01,'')
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

    fm_size = FLAGS.image_size//8

    train_iter = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(10):
        for data in training_generator:
            optimizer.zero_grad()

            image_crops, labels, crop_dims = data
            features = []
            dists = []
            for c in range(FLAGS.num_crops):
                if FLAGS.feed_labels:
                    features.append(model(image_crops[c].to('cuda'), labels=labels[c].to('cuda')).reshape(FLAGS.batch_size, fm_size, fm_size, FLAGS.output_dim))
                else:
                    features.append(model(image_crops[c].to('cuda')).reshape(FLAGS.batch_size, fm_size, fm_size, FLAGS.output_dim))

                if FLAGS.dist_func == 'euc':
                    dists.append(((features[c].unsqueeze(-2) - model.prototypes)**2).mean(dim=-1).reshape(FLAGS.batch_size, fm_size, fm_size, FLAGS.num_prototypes))
                elif FLAGS.dist_func == 'dot':
                    dists.append(-(features[c].unsqueeze(-2) * model.prototypes).mean(dim=-1).reshape(FLAGS.batch_size, fm_size, fm_size, FLAGS.num_prototypes))
                elif FLAGS.dist_func == 'cosine':
                    dists.append(-(F.normalize(features[c].unsqueeze(-2),dim=-1) * F.normalize(model.prototypes,dim=-1)).sum(dim=-1) \
                        .reshape(FLAGS.batch_size, fm_size, fm_size, FLAGS.num_prototypes))

            contrastive_loss = 0.
            for c in range(FLAGS.num_crops-1):
                for k in range(c+1, FLAGS.num_crops):
                    dists_crop_a, dists_crop_b = model.match_crops(dists[c], dists[k], [crop_dims[c], crop_dims[k]])
                    dists_crop_a = dists_crop_a.reshape(-1, FLAGS.num_prototypes)
                    dists_crop_b = dists_crop_b.reshape(-1, FLAGS.num_prototypes)
                    
                    Q_a = sinkhorn_knopp(-dists_crop_a)
                    Q_b = sinkhorn_knopp(-dists_crop_b)
                    loss_a = F.cross_entropy(-dists_crop_a/FLAGS.temperature, Q_b)
                    loss_b = F.cross_entropy(-dists_crop_b/FLAGS.temperature, Q_a)
                    contrastive_loss += loss_a + loss_b

            label = labels[0].reshape(-1).to('cuda')
            preds = F.interpolate(model.class_pred(F.one_hot(torch.argmin(dists[0], dim=-1), FLAGS.num_prototypes).to(torch.float32)).movedim(3,1), \
                                    size=FLAGS.image_size,mode='nearest').movedim(1,3).reshape(-1, 27)
            pred_loss = F.cross_entropy(preds, label)

            loss = pred_loss + contrastive_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = (preds.argmax(dim=-1) == label).to(torch.float32).mean()

                acc_clust = (dists_crop_a.argmin(dim=-1) == dists_crop_b.argmin(dim=-1)).float().mean()

                avg_clusters_per_image = Q_a.reshape(FLAGS.batch_size, -1, FLAGS.num_prototypes).sum(dim=1)
                avg_clusters_per_image = (avg_clusters_per_image > 0).float().sum(dim=1).mean()

                avg_samples_per_cluster = Q_a.sum(dim=0)
                total_samples = avg_samples_per_cluster.sum()
                max_samples = avg_samples_per_cluster.max() / total_samples
                min_samples = avg_samples_per_cluster.min() / total_samples
                tenth_highest = avg_samples_per_cluster.topk(10)[0][-1] / total_samples
                tenth_lowest = -(-avg_samples_per_cluster).topk(10)[0][-1] / total_samples

            log_dict = {"Epoch": epoch, "Iter": train_iter, "Total Loss": loss.item(), "Pred Loss": pred_loss.item(), "Cluster Acc": acc_clust.item(), \
                        "Acc": acc.item(), "Contrastive Loss": contrastive_loss.item(), "Num Clusters": avg_clusters_per_image.item(), \
                        "Most Samples": max_samples.item(), "Least Samples": min_samples.item(), "10th Highest": tenth_highest.item(), "10th Lowest": tenth_lowest.item()}
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