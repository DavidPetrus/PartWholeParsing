import numpy as np
import cv2
import torch
import torch.nn.functional as F
#from torchvision import ColorJitter
import glob
import datetime
import time
import copy
import pickle as pkl

from dataloader import ADE20k_2017, ADE_Challenge, Coco, CelebA
from model import ImageParser
from utils import sinkhorn_knopp, unnormalize, display_label, vic_reg
import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('data_dir', '/mnt/lustre/users/dvanniekerk1', '')
flags.DEFINE_string('dataset','CelebA','ADE_Partial, ADE_Full, coco')
flags.DEFINE_bool('feed_labels',False,'')
flags.DEFINE_bool('save_labels',False,'')
flags.DEFINE_integer('batch_size',32,'')
flags.DEFINE_float('lr',0.0003,'')
flags.DEFINE_integer('num_workers',8,'')
flags.DEFINE_integer('image_size',224,'')
flags.DEFINE_integer('num_crops',2,'')
flags.DEFINE_float('min_crop',0.55,'Height/width size of crop')
flags.DEFINE_float('teacher_momentum', 0.98, '')
flags.DEFINE_float('entropy_reg', 0.1, '')
flags.DEFINE_float('mean_max_coeff', 0.5, '')

flags.DEFINE_integer('num_output_classes', 28, '')
flags.DEFINE_float('entropy_temp', 0.05, '')
flags.DEFINE_float('student_temp', 0.1, '')
flags.DEFINE_float('teacher_temp', 0.04, '')
flags.DEFINE_integer('depth', 3, '')
flags.DEFINE_integer('embd_dim', 384, '')
flags.DEFINE_integer('output_dim', 32, '')

flags.DEFINE_float('aug_strength', 0.7, '')


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

    elif FLAGS.dataset == 'CelebA':
        training_set = CelebA("train")
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

        validation_set = CelebA("val")
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)


    student = ImageParser("vit_small")
    teacher = ImageParser("vit_small")

    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=FLAGS.lr)

    fm_size = FLAGS.image_size//8
    center = torch.zeros(1, FLAGS.output_dim).to('cuda')

    entropy_norm = -np.log(1/FLAGS.output_dim)

    train_iter = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(1):
        for data in training_generator:
            optimizer.zero_grad()

            image_crops, labels, crop_dims = data
            #print(crop_dims)
            for c in range(FLAGS.num_crops):
                image_crops[c] = image_crops[c].to('cuda')
                #labels[c] = labels[c].to('cuda')

            proj_feats_s = []
            proj_feats_t = []
            entropy_reg = 0.
            for c in range(FLAGS.num_crops):
                if FLAGS.feed_labels:
                    proj_feat_s = student(image_crops[c], labels=labels[c]) # bs, l, c
                    proj_feat_t = teacher(image_crops[c], labels=labels[c])
                else:
                    proj_feat_s = student(image_crops[c])
                    proj_feat_t = teacher(image_crops[c])

                proj_feat_s = proj_feat_s - FLAGS.mean_max_coeff * (proj_feat_s.mean(dim=1, keepdim=True) + proj_feat_s.max(dim=1, keepdim=True)[0])
                proj_feat_t = proj_feat_t - FLAGS.mean_max_coeff * (proj_feat_t.mean(dim=1, keepdim=True) + proj_feat_t.max(dim=1, keepdim=True)[0]) 

                proj_feats_s.append(proj_feat_s)
                proj_feats_t.append(proj_feat_t)

                entropy_reg += -torch.log(F.softmax(proj_feat_s/FLAGS.entropy_temp, dim=-1).mean(dim=1)).mean()

            contrastive_loss = 0.
            for s in range(FLAGS.num_crops):
                for t in range(FLAGS.num_crops):
                    if s == t:
                        continue

                    feat_crop_s, feat_crop_t = student.match_crops(proj_feats_s[s], proj_feats_t[t], [crop_dims[s], crop_dims[t]])
                    feat_crop_s = feat_crop_s.reshape(-1, FLAGS.output_dim)
                    feat_crop_t = feat_crop_t.reshape(-1, FLAGS.output_dim)

                    contrastive_loss += F.cross_entropy(feat_crop_s/FLAGS.student_temp, F.softmax(feat_crop_t/FLAGS.teacher_temp, dim=-1).detach())

            label = labels[0].reshape(-1)
            preds = F.interpolate(student.class_pred(feat_s.detach()).reshape(FLAGS.batch_size, fm_size, fm_size, FLAGS.num_output_classes).movedim(3,1), \
                                    size=FLAGS.image_size,mode='bilinear').movedim(1,3).reshape(-1, FLAGS.num_output_classes)
            pred_loss = F.cross_entropy(preds[label != FLAGS.num_output_classes-1], label[label != FLAGS.num_output_classes-1])

            loss = pred_loss + contrastive_loss + FLAGS.entropy_reg*entropy_reg

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                #m = momentum_schedule[it]  # momentum parameter
                m = FLAGS.teacher_momentum
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

                acc = (preds.argmax(dim=-1)[label != FLAGS.num_output_classes-1] == label[label != FLAGS.num_output_classes-1]).to(torch.float32).mean()

                acc_clust = (feat_crop_s.argmax(dim=-1) == feat_crop_t_center.argmax(dim=-1)).float().mean()

                max_feat = proj_feat_s.argmax(dim=-1)
                uniq_cat, output_counts = torch.unique(max_feat, return_counts = True)
                most_freq_frac = output_counts.max() / output_counts.sum()

                max_feat_img = proj_feat_s[0].argmax(dim=-1)
                _, output_counts_img = torch.unique(max_feat_img, return_counts = True)
                most_freq_frac_img = output_counts_img.max() / output_counts_img.sum()

                if train_iter > 0:
                    label = label.reshape(FLAGS.batch_size, 1, FLAGS.image_size, FLAGS.image_size)[:1][0,0].cpu().long().numpy()
                    lab_disp = display_label(label)
                    cv2.imwrite(f'images/{FLAGS.exp}_{train_iter}_target_lab1.png', lab_disp)
                    img = (255*unnormalize(image_crops[0][0])).long().movedim(0,2).cpu().numpy()[:,:,::-1]
                    cv2.imwrite(f'images/{FLAGS.exp}_{train_iter}_crop1.png', img)

                    np.save(f'images/{FLAGS.exp}_{train_iter}_student.npy', proj_feats_s[0].cpu().numpy())
                    np.save(f'images/{FLAGS.exp}_{train_iter}_teacher.npy', proj_feats_t[1].cpu().numpy())


                if train_iter > 3030:
                    exit()

            log_dict = {"Epoch": epoch, "Iter": train_iter, "Total Loss": loss.item(), "Pred Loss": pred_loss.item(), "Cluster Acc": acc_clust.item(), \
                        "Acc": acc.item(), "Contrastive Loss": contrastive_loss.item(), "Num Categories": output_counts.shape[0], "Num Categories Image": output_counts_img.shape[0], \
                        "Entropy Reg": entropy_reg.item(), "Most Freq Cat": most_freq_frac.item(), "Most Freq Cat Image": most_freq_frac_img.item()}
            
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