import numpy as np
import cv2
import torch
import torch.nn.functional as F
import random
torch.manual_seed(66)
np.random.seed(66)
random.seed(66)
#from torchvision import ColorJitter
import glob
import datetime
import time
import copy
import pickle as pkl
import os

from dataloader import ADE20k_2017, ADE_Challenge, Coco, CelebA, Cityscapes
from model import ImageParser
from utils import unnormalize, display_label, display_mask, calc_mIOU, normalize_feature_maps, calc_hungarian_mIOU
import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('data_dir', '/mnt/lustre/users/dvanniekerk1', '')
flags.DEFINE_string('dataset','cityscapes','ADE_Partial, ADE_Full, coco')
flags.DEFINE_integer('num_epochs', 60, '')
flags.DEFINE_bool('save_images',True,'')
flags.DEFINE_bool('instance_seg', False, '')
flags.DEFINE_string('backbone','dinov1','dinov1, dinov2')
flags.DEFINE_string('seg_layers','attn','attn or conv')
flags.DEFINE_bool('train_dinov1', False, '')
flags.DEFINE_bool('train_dinov2', False, '')
flags.DEFINE_bool('train_dino_resnet', True, '')
flags.DEFINE_integer('batch_size',32,'')
flags.DEFINE_float('lr',0.0003,'')
flags.DEFINE_integer('num_workers',8,'')
flags.DEFINE_integer('image_size',224,'')
flags.DEFINE_integer('eval_size',336,'')
flags.DEFINE_integer('num_crops',2,'')
flags.DEFINE_float('min_crop',0.55,'Height/width size of crop')
flags.DEFINE_float('teacher_momentum', 0.99, '')
flags.DEFINE_float('cont_coeff',0.3,'')
flags.DEFINE_float('score_coeff',3.,'')
flags.DEFINE_float('entropy_reg', 0., '')
flags.DEFINE_float('mean_max_coeff', 0.75, '')
flags.DEFINE_string('norm_type', 'mean_max', 'mean_max, mean_std, mean')
flags.DEFINE_integer('miou_bs',1,'')
flags.DEFINE_integer('sem_width', 1024, '')

flags.DEFINE_integer('num_output_classes', 27, '')
flags.DEFINE_float('entropy_temp', 0.05, '')
flags.DEFINE_float('student_temp', 0.1, '')
flags.DEFINE_float('teacher_temp', 0.04, '')
flags.DEFINE_integer('depth', 1, '')
flags.DEFINE_integer('proj_depth',1,'')
flags.DEFINE_integer('kernel_size', 3, '')
flags.DEFINE_integer('embd_dim', 384, '')
flags.DEFINE_float('dice_factor',3,'')
flags.DEFINE_integer('outp_dim2',0,'')
flags.DEFINE_integer('output_dim', 128, '')

flags.DEFINE_bool('linear_score',False,'')

flags.DEFINE_bool('student_eval', False, '')

flags.DEFINE_float('aug_strength', 0.7, '')
flags.DEFINE_bool('flip_image', True, '')
flags.DEFINE_float('fm_noise', 0.1, '')
flags.DEFINE_float('patch_masking', 0., '')
flags.DEFINE_float('dropout',0., '')
flags.DEFINE_float('weight_decay',0.01,'')


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

    elif FLAGS.dataset == 'cityscapes':
        training_set = Cityscapes("train")
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

        validation_set = Cityscapes("val")
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)


    student = ImageParser("vit_small", dropout=FLAGS.dropout)
    teacher = ImageParser("vit_small")

    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    if FLAGS.save_images:
        if not os.path.exists('images/'+FLAGS.exp):
            os.mkdir('images/'+FLAGS.exp)

    train_iter = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(FLAGS.num_epochs):

        val_iter = 0
        val_miou, val_pix_acc = 0.,0.
        with torch.no_grad():
            student.eval()
            teacher.eval()
            for data in validation_generator:

                images, labels = data
                images = images.to('cuda')
                labels = labels.to('cuda')

                if FLAGS.student_eval:
                    proj_feat = student(images, val=True)
                    #_, cluster_preds = student.cluster_lookup(seg_feat)
                else:
                    proj_feat = teacher(images, val=True)
                    #_, cluster_preds = teacher.cluster_lookup(seg_feat)

                proj_feat = normalize_feature_maps(proj_feat)

                label = labels[:FLAGS.miou_bs].long().squeeze(1)
                #cluster_preds = F.upsample(cluster_preds[:FLAGS.miou_bs], scale_factor=4)
                proj_feat_up = F.upsample(proj_feat[:FLAGS.miou_bs], scale_factor=8)

                #pred_cluster_acc = (cluster_preds.argmax(dim=1)[label >= 0] == label.long()[label >= 0]).to(torch.float32).mean()
                #acc_clust = (feat_crop_s.argmax(dim=-1) == feat_crop_t.argmax(dim=-1)).float().mean()

                label[label < 0] = FLAGS.num_output_classes
                label_one_hot = F.one_hot(label, FLAGS.num_output_classes+1)[:,:,:,:-1].movedim(3,1) # bs,nc,h,w
                #pred_clust_miou = calc_mIOU(F.one_hot(cluster_preds.argmax(dim=1), FLAGS.num_output_classes).movedim(3,1), label_one_hot)
                #proj_clust_miou = calc_mIOU(F.one_hot(proj_feat_up.argmax(dim=1), FLAGS.output_dim).movedim(3,1), label_one_hot)

                if FLAGS.outp_dim2 > 0:
                    proj_feat_cat = torch.cat([F.one_hot(proj_feat_up[:,:FLAGS.output_dim].argmax(dim=1), FLAGS.output_dim), \
                               F.one_hot(proj_feat_up[:,FLAGS.output_dim:].argmax(dim=1), FLAGS.outp_dim2)], dim=3).movedim(3,1)
                    hungarian_mIOU, pixel_acc = calc_hungarian_mIOU(proj_feat_cat, label_one_hot)
                else:
                    hungarian_mIOU, pixel_acc = calc_hungarian_mIOU(F.one_hot(proj_feat_up.argmax(dim=1), FLAGS.output_dim).movedim(3,1), label_one_hot)
                

                val_iter += 1
                val_miou += hungarian_mIOU
                val_pix_acc += pixel_acc

                if val_iter > 3 and epoch==0:
                    break

            if FLAGS.save_images:
                label = labels[0].cpu().long().numpy()
                lab_disp = display_label(label.squeeze())
                cv2.imwrite(f'images/{FLAGS.exp}/{epoch}_label.png', lab_disp)

                img = (255*unnormalize(images[0])).long().movedim(0,2).cpu().numpy()[:,:,::-1]
                cv2.imwrite(f'images/{FLAGS.exp}/{epoch}_crop.png', img)

                mask = proj_feat_up[0].argmax(dim=0)
                mask_disp = display_mask(mask.cpu().numpy())
                cv2.imwrite(f'images/{FLAGS.exp}/{epoch}_mask.png', mask_disp)


            log_dict = {"Epoch": epoch, "Val Hungarian mIOU": val_miou/val_iter, "Val Pixel Acc": val_pix_acc/val_iter}
            print(log_dict)

            wandb.log(log_dict)

        student.train()
        teacher.train()
        for data in training_generator:
            student_crops, labels, crop_dims = data
            for c in range(FLAGS.num_crops):
                student_crops[c] = student_crops[c].to('cuda')
                #teacher_crops[c] = teacher_crops[c].to('cuda')
                labels[c] = labels[c].to('cuda')

            proj_feats_s = []
            proj_feats_t = []
            dino_feats = []
            seg_feats = []
            scores_s = []
            scores_t = []
            for c in range(FLAGS.num_crops):
                unnorm_feat_s = student(student_crops[c], student=True)
                unnorm_feat_t = teacher(student_crops[c])

                proj_feat_s = normalize_feature_maps(unnorm_feat_s)
                proj_feat_t = normalize_feature_maps(unnorm_feat_t)

                proj_feats_s.append(proj_feat_s) # bs,no,h,w
                proj_feats_t.append(proj_feat_t) # bs,no,h,w
                #seg_feats.append(seg_feat) # bs,c,h,w
                #dino_feats.append(dino_feat) # bs,c,h,w

                #entropy_reg += -torch.log(F.softmax(proj_feat_s/FLAGS.entropy_temp, dim=-1).mean(dim=(2,3))).mean()

                #preds_s = F.softmax(proj_feat_s/FLAGS.student_temp, dim=1) # bs,no,h,w
                #scores_s.append(student.obtain_scores(preds_s, unnorm_feat_s)) # bs,no
                #preds_t = F.softmax(proj_feat_t/FLAGS.teacher_temp, dim=1) # bs,no,h,w
                #scores_t.append(teacher.obtain_scores(preds_t, unnorm_feat_s)) # bs,no


            contrastive_loss = 0.
            dice_loss = 0.
            score_loss = 0.
            for s in range(FLAGS.num_crops):
                for t in range(FLAGS.num_crops):
                    if s == t:
                        continue

                    feat_crop_s, feat_crop_t = student.match_crops(proj_feats_s[s], proj_feats_t[t], [crop_dims[s], crop_dims[t]])

                    if FLAGS.outp_dim2 > 0:
                        target = F.softmax(feat_crop_t[:,-FLAGS.outp_dim2:]/FLAGS.teacher_temp, dim=1).detach()
                        contrastive_loss += F.cross_entropy(feat_crop_s[:,-FLAGS.outp_dim2:]/FLAGS.student_temp, target, reduction='mean')

                        preds = F.softmax(feat_crop_s[:,-FLAGS.outp_dim2:]/FLAGS.student_temp, dim=1)
                        present_cats = target.mean(dim=(2,3)) > 0.001 # bs, c
                        dice_term = 2*(preds*target).sum(dim=(2,3))/(preds.sum(dim=(2,3)) + target.sum(dim=(2,3)) + 0.0001) # bs,c
                        dice_loss += (1 - dice_term[present_cats].mean()) / FLAGS.dice_factor

                        feat_crop_s = feat_crop_s[:,:-FLAGS.outp_dim2]
                        feat_crop_t = feat_crop_t[:,:-FLAGS.outp_dim2]

                    target = F.softmax(feat_crop_t/FLAGS.teacher_temp, dim=1).detach()

                    preds = F.softmax(feat_crop_s/FLAGS.student_temp, dim=1)
                    present_cats = target.mean(dim=(2,3)) > (2 / (target.shape[2]*target.shape[3])) # bs, c
                    if FLAGS.instance_seg:
                        numerator = (preds.unsqueeze(2) * target.unsqueeze(1)).sum(dim=(-1,-2)) # bs, num_targets, num_preds
                        denominator = (preds.unsqueeze(2)**2 + target.unsqueeze(1)**2).sum(dim=(-1,-2)) # bs, num_targets, num_preds
                        overlaps = 2*numerator / (denominator + 0.0001)
                        dice_term, max_idxs = overlaps.max(dim=2) # bs,num_targets

                        student_matches = torch.gather(preds, 1, max_idxs.reshape(max_idxs.shape[0],max_idxs.shape[1],1,1).tile(1,1,preds.shape[2],preds.shape[3]))
                        contrastive_loss += F.cross_entropy(student_matches/FLAGS.student_temp, target, reduction='mean')
                    else:
                        contrastive_loss += F.cross_entropy(feat_crop_s/FLAGS.student_temp, target, reduction='mean')
                        dice_term = 2*(preds*target).sum(dim=(2,3))/((preds**2).sum(dim=(2,3)) + (target**2).sum(dim=(2,3)) + 0.0001) # bs,c
                    
                    #crop_scores = scores_t[t][present_cats].detach()
                    #dice_loss += 1 - (dice_term[present_cats] * crop_scores).sum() / crop_scores.sum()
                    dice_loss += 1 - dice_term[present_cats].mean()
                    
                    #score_loss += F.mse_loss(scores_s[s][present_cats], dice_term[present_cats].detach())


            #dino_loss, dino_preds = student.cluster_lookup(dino_feats[0].detach(), dino_cluster=True) # _, bs, num_classes, h, w
            #cluster_loss, cluster_preds = student.cluster_lookup(seg_feats[0].detach()) # _, bs, num_classes, h, w

            loss = dice_loss + FLAGS.cont_coeff * contrastive_loss

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            with torch.no_grad():
                # Update teacher parameters
                #m = momentum_schedule[it]  # momentum parameter
                m = FLAGS.teacher_momentum
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

                # Compute mIOU between student and teacher predictions
                feat_s_argmax = feat_crop_s.argmax(dim=1)
                feat_t_argmax = feat_crop_t.argmax(dim=1)
                acc_clust = (feat_s_argmax == feat_t_argmax).float().mean()

                intersection = torch.logical_and(F.one_hot(feat_s_argmax, FLAGS.output_dim), F.one_hot(feat_t_argmax, FLAGS.output_dim)).sum(dim=(1,2))
                union = torch.logical_or(F.one_hot(feat_s_argmax, FLAGS.output_dim), F.one_hot(feat_t_argmax, FLAGS.output_dim)).sum(dim=(1,2))
                iou = intersection / (union + 0.001) # bs, output_dim
                present_cats = (union > 0) # bs, output_dim
                cluster_mIOU = (iou * present_cats.float()).sum(dim=1) / present_cats.float().sum(dim=1) # bs
                cluster_mIOU = cluster_mIOU.mean()

                # Compute mIOU between predictions and labels
                #label = F.interpolate(labels[0], size=FLAGS.image_size//2, mode='nearest').long().squeeze() # bs,h,w
                label = labels[0][:FLAGS.miou_bs].long().squeeze(1)
                label[label < 0] = FLAGS.num_output_classes
                #cluster_preds = F.upsample(cluster_preds[:FLAGS.miou_bs], scale_factor=4)
                #dino_preds = F.upsample(dino_preds[:FLAGS.miou_bs], scale_factor=14)
                proj_feat_up = F.upsample(proj_feats_s[0][:FLAGS.miou_bs], scale_factor=8)

                #pred_cluster_acc = (cluster_preds.argmax(dim=1)[label >= 0] == label.long()[label >= 0]).to(torch.float32).mean()
                #dino_acc = (dino_preds.argmax(dim=1)[label >= 0] == label.long()[label >= 0]).to(torch.float32).mean()

                label_one_hot = F.one_hot(label, FLAGS.num_output_classes+1)[:,:,:,:-1].movedim(3,1) # bs,nc,h,w
                #pred_clust_miou = calc_mIOU(F.one_hot(cluster_preds.argmax(dim=1), FLAGS.num_output_classes).movedim(3,1), label_one_hot)
                #dino_clust_miou = calc_mIOU(F.one_hot(dino_preds.argmax(dim=1), FLAGS.num_output_classes).movedim(3,1), label_one_hot)
                #proj_clust_miou = calc_mIOU(F.one_hot(proj_feat_up.argmax(dim=1), FLAGS.output_dim).movedim(3,1), label_one_hot)
                if FLAGS.outp_dim2 > 0:
                    proj_feat_cat = torch.cat([F.one_hot(proj_feat_up[:,:FLAGS.output_dim].argmax(dim=1), FLAGS.output_dim), \
                        F.one_hot(proj_feat_up[:,FLAGS.output_dim:].argmax(dim=1), FLAGS.outp_dim2)], dim=3).movedim(3,1)
                    hungarian_mIOU, pixel_acc = calc_hungarian_mIOU(proj_feat_cat, label_one_hot)
                else:
                    hungarian_mIOU, pixel_acc = calc_hungarian_mIOU(F.one_hot(proj_feat_up.argmax(dim=1), FLAGS.output_dim).movedim(3,1), label_one_hot)

                max_feat = proj_feat_t.argmax(dim=1)
                uniq_cat, output_counts = torch.unique(max_feat, return_counts = True)
                most_freq_frac = output_counts.max() / output_counts.sum()

                max_feat_img = proj_feat_t[0].argmax(dim=0)
                _, output_counts_img = torch.unique(max_feat_img, return_counts = True)
                most_freq_frac_img = output_counts_img.max() / output_counts_img.sum()

            log_dict = {"Epoch": epoch, "Iter": train_iter, "Total Loss": loss.item(), "Cluster Acc": acc_clust.item(), "Dice Loss": dice_loss.item(), \
                        "Contrastive_mIOU": cluster_mIOU.item(), "Hungarian_mIOU": hungarian_mIOU.item(), "Pixel_Acc": pixel_acc.item(), \
                        "Contrastive Loss": contrastive_loss.item(), "Num Categories": output_counts.shape[0], "Num Categories Image": output_counts_img.shape[0], \
                        "Most Freq Cat": most_freq_frac.item(), "Most Freq Cat Image": most_freq_frac_img.item()}
            
            if train_iter % 10 == 0:
                print(log_dict)

            train_iter += 1

            wandb.log(log_dict)


        

        



if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)