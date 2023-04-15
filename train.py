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

from dataloader import ADE20k_2017, ADE_Challenge, Coco
from model import ImageParser
from utils import sinkhorn_knopp, unnormalize, display_label, vic_reg
from matcher import HungarianMatcher, calc_loss

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('data_dir', '/mnt/lustre/users/dvanniekerk1', '')
flags.DEFINE_string('dataset','coco','ADE_Partial, ADE_Full, coco')
flags.DEFINE_bool('feed_labels',False,'')
flags.DEFINE_bool('save_labels',False,'')
flags.DEFINE_integer('batch_size',32,'')
flags.DEFINE_float('lr',0.0003,'')
flags.DEFINE_integer('num_workers',8,'')
flags.DEFINE_integer('image_size',224,'')
flags.DEFINE_integer('num_crops',2,'')
flags.DEFINE_float('min_crop',0.55,'Height/width size of crop')
flags.DEFINE_float('teacher_momentum', 0.98, '')
flags.DEFINE_integer('num_masks', 100,'')
flags.DEFINE_integer('num_decoder_layers', 3, '')
flags.DEFINE_integer('output_patch_size', 4, '')
flags.DEFINE_float('mask_reg_coeff',1.,'')
flags.DEFINE_float('mask_cover_coeff',0.03,'')
flags.DEFINE_float('mask_cover_temp',0.1,'')
flags.DEFINE_float('alpha',0.5,'')
flags.DEFINE_float('gamma',0.,'')
flags.DEFINE_float('mean_max_sub', 0.5, '')

flags.DEFINE_integer('num_output_classes', 28, '')
flags.DEFINE_float('student_temp',0.2,'')
flags.DEFINE_float('teacher_temp', 0.03, '')
flags.DEFINE_integer('depth', 3, '')
flags.DEFINE_integer('embd_dim', 384, '')

flags.DEFINE_float('aug_strength', 0.7, '')
flags.DEFINE_float('clip_grad', 300, '')


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


    student = ImageParser("vit_small")
    teacher = ImageParser("vit_small")

    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=FLAGS.lr)

    matcher = HungarianMatcher()

    train_iter = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(1):
        for data in training_generator:
            optimizer.zero_grad()

            image_crops, labels, crop_dims = data
            for c in range(FLAGS.num_crops):
                image_crops[c] = image_crops[c].to('cuda')
                labels[c] = labels[c].to('cuda')

            all_masks_s = []
            all_masks_t = []
            reg_loss = 0.
            mask_cover_loss = 0.
            for c in range(FLAGS.num_crops):
                if FLAGS.feed_labels:
                    feat_s, proj_feat_s = student(image_crops[c], labels=labels[c])
                    feat_t, proj_feat_t = teacher(image_crops[c], labels=labels[c])
                else:
                    masks_s = student(image_crops[c]) # bs, num_masks, h, w
                    masks_t = teacher(image_crops[c])

                # Normalize masks to ensure each mask has some positive pixels
                masks_s = masks_s - FLAGS.mean_max_sub * (masks_s.mean(dim=(2,3), keepdim=True) + masks_s.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(-1))
                masks_t = masks_t - FLAGS.mean_max_sub * (masks_t.mean(dim=(2,3), keepdim=True) + masks_t.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(-1))

                mask_cover_loss += -F.log_softmax((F.relu(masks_s).sum(dim=1).flatten(1) + 1).log2() / FLAGS.mask_cover_temp, dim=1).mean()
                num_pos_per_pixel = (masks_s > 0).sum(dim=1) # bs,h,w
                min_num_pos = num_pos_per_pixel.flatten(1).min(dim=1)[0].float().mean()
                max_num_pos = num_pos_per_pixel.flatten(1).max(dim=1)[0].float().mean()
                frac_covered = (num_pos_per_pixel > 0).float().mean(dim=(1,2))

                reg_loss += masks_s.sigmoid().mean()

                all_masks_s.append(masks_s)
                all_masks_t.append(masks_t)

            dice_loss, focal_loss = 0., 0.
            for s in range(FLAGS.num_crops):
                for t in range(FLAGS.num_crops):
                    if s == t:
                        continue

                    mask_crop_s, mask_crop_t = student.match_crops(all_masks_s[s], all_masks_t[t], [crop_dims[s], crop_dims[t]]) # bs,h,w,num_masks
                    mask_crop_s = mask_crop_s.movedim(3,1) / FLAGS.student_temp # bs,num_masks,h,w
                    mask_crop_t = mask_crop_t.movedim(3,1) / FLAGS.teacher_temp
                    matched_indices, num_masks_t = matcher(mask_crop_s, mask_crop_t)
                    losses = calc_loss(mask_crop_s, mask_crop_t.sigmoid(), matched_indices)

                    dice_loss += losses["loss_dice"]
                    focal_loss += losses["loss_mask"]


            '''label = labels[0].reshape(-1)
            preds = F.interpolate(student.class_pred(feat_s.detach()).reshape(FLAGS.batch_size, fm_size, fm_size, FLAGS.num_output_classes).movedim(3,1), \
                                    size=FLAGS.image_size,mode='bilinear').movedim(1,3).reshape(-1, FLAGS.num_output_classes)
            pred_loss = F.cross_entropy(preds[label != FLAGS.num_output_classes-1], label[label != FLAGS.num_output_classes-1])'''

            loss = 0.05*dice_loss + focal_loss + FLAGS.mask_reg_coeff*reg_loss + FLAGS.mask_cover_coeff*mask_cover_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), FLAGS.clip_grad)
            optimizer.step()

            with torch.no_grad():
                #m = momentum_schedule[it]  # momentum parameter
                m = FLAGS.teacher_momentum
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

                frac_pos_pix = (masks_s > 0).float().mean(dim=(2,3), keepdim=True) # bs, num_masks, 1, 1
                min_pos_pix = frac_pos_pix.min(dim=1)[0].mean()
                max_pos_pix = frac_pos_pix.max(dim=1)[0].mean()
                mean_pos_pix = frac_pos_pix.mean()

                '''acc = (preds.argmax(dim=-1)[label != FLAGS.num_output_classes-1] == label[label != FLAGS.num_output_classes-1]).to(torch.float32).mean()

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
                    np.save(f'images/{FLAGS.exp}_{train_iter}_teacher.npy', proj_feats_t[1].cpu().numpy())'''


            log_dict = {"Epoch": epoch, "Iter": train_iter, "Total Loss": loss.item(), "Dice Loss": dice_loss.item(), "Focal Loss": focal_loss.item(), "Mask Reg Loss": reg_loss.item(), \
                        "Min Pos Frac": min_pos_pix.item(), "Mean Pos Frac": mean_pos_pix.item(), "Max Pos Frac": max_pos_pix.item(), "Num Masks Teacher": num_masks_t, \
                        "Grad Norm": grad_norm.item(), "Mask Cover Loss": mask_cover_loss.item(), "Min Masks Pixel": min_num_pos.item(), \
                        "Mean Masks Pixel": num_pos_per_pixel.float().mean().item(), "Max Masks Pixel": max_num_pos.item(), "Min Frac Covered": frac_covered.min().item(), \
                        "Mean Frac Covered": frac_covered.mean().item(), "Max Frac Covered": frac_covered.max().item()}
            
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