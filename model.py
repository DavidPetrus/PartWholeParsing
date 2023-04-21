import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm

import dino.vision_transformer as vits
from utils import find_clusters

from absl import flags

FLAGS = flags.FLAGS

class ImageParser(nn.Module):

    def __init__(self, arch):
        super(ImageParser, self).__init__()

        if FLAGS.use_dino:
            self.model = vits.__dict__[arch](
                patch_size=8,
                num_classes=0)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval().cuda()
            self.dropout = nn.Dropout2d(p=.1)

            if arch == "vit_small":
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            elif arch == "vit_base":
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model = timm.create_model('resnest26d', features_only=True, pretrained=False, out_indices=(2,)).to('cuda')
            self.proj = nn.Conv2d(512, FLAGS.embd_dim, kernel_size=1).to('cuda')

        segment_net = []
        segment_net.extend([nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=FLAGS.kernel_size, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim), nn.ReLU()])
        segment_net.extend([nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=FLAGS.kernel_size, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim), nn.ReLU()])
        if FLAGS.output_stride == 2:
            segment_net.append(nn.Upsample(scale_factor=2))
            segment_net.extend([nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim//2, kernel_size=FLAGS.kernel_size, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim//2), nn.ReLU()])
            segment_net.append(nn.Upsample(scale_factor=2))
            segment_net.extend([nn.Conv2d(FLAGS.embd_dim//2, FLAGS.embd_dim//2, kernel_size=FLAGS.kernel_size, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim//2)])

            self.segment_net = nn.Sequential(*segment_net).to('cuda')
            self.proj_layer = nn.Conv2d(FLAGS.embd_dim//2, FLAGS.output_dim, kernel_size=1).to('cuda')
            self.clusters = torch.nn.Parameter(torch.randn(FLAGS.num_output_classes, FLAGS.embd_dim//2)).to('cuda')
        elif FLAGS.output_stride == 4:
            #segment_net.extend([nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=FLAGS.kernel_size, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim), nn.ReLU()])
            segment_net.append(nn.Upsample(scale_factor=2))
            segment_net.extend([nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=FLAGS.kernel_size, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim)])

            self.segment_net = nn.Sequential(*segment_net).to('cuda')
            self.proj_layer = nn.Conv2d(FLAGS.embd_dim, FLAGS.output_dim, kernel_size=1).to('cuda')
            self.clusters = torch.nn.Parameter(torch.randn(FLAGS.num_output_classes, FLAGS.embd_dim)).to('cuda')
        elif FLAGS.output_stride == 8:
            segment_net.extend([nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=FLAGS.kernel_size, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim)])

            self.segment_net = nn.Sequential(*segment_net).to('cuda')
            self.proj_layer = nn.Conv2d(FLAGS.embd_dim, FLAGS.output_dim, kernel_size=1).to('cuda')
            self.clusters = torch.nn.Parameter(torch.randn(FLAGS.num_output_classes, FLAGS.embd_dim)).to('cuda')

        self.dino_clusters = torch.nn.Parameter(torch.randn(FLAGS.num_output_classes, FLAGS.embd_dim)).to('cuda')

    def cluster_lookup(self, x, dino_cluster=False):
        normed_clusters = F.normalize(self.dino_clusters, dim=1) if dino_cluster else F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)

        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), FLAGS.num_output_classes).permute(0, 3, 1, 2).to(torch.float32)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        return cluster_loss, cluster_probs
    

    def forward(self, x, val=False):
        self.model.eval()
        with torch.set_grad_enabled(not FLAGS.use_dino):
            # get dino activations
            dino_feat = self.model(x)
            if val:
                dino_feat = dino_feat[:,1:].reshape(FLAGS.batch_size//2, FLAGS.eval_size//8, FLAGS.eval_size//8, FLAGS.embd_dim).movedim(3,1) # bs,c,h,w
            else:
                dino_feat = dino_feat[:,1:].reshape(FLAGS.batch_size, FLAGS.image_size//8, FLAGS.image_size//8, FLAGS.embd_dim).movedim(3,1) # bs,c,h,w

        feat = self.segment_net(dino_feat) # bs,c,h,w
        masks = self.proj_layer(feat).movedim(1,3) # bs,h,w,c

        return masks, feat, dino_feat

    def match_crops(self, sims_a, sims_b, crop_dims):
        #sims_a = sims_a.reshape(FLAGS.batch_size, self.fm_size, self.fm_size, FLAGS.output_dim)
        #sims_b = sims_b.reshape(FLAGS.batch_size, self.fm_size, self.fm_size, FLAGS.output_dim)
        b,fm_size,_,c = sims_a.shape
        if crop_dims[1][2] > crop_dims[0][2]:
            l_map = sims_b
            s_map = sims_a
            l_dims = crop_dims[1]
            s_dims = crop_dims[0]
        else:
            l_map = sims_a
            s_map = sims_b
            l_dims = crop_dims[0]
            s_dims = crop_dims[1]

        ratio = l_dims[2]/s_dims[2]

        fm_scale_l = fm_size/l_dims[2]
        fm_scale_s = fm_size/s_dims[2]

        overlap_dims = (max(s_dims[0], l_dims[0]), \
                        max(s_dims[1], l_dims[1]), \
                        min(s_dims[0]+s_dims[2], l_dims[0]+l_dims[2]), \
                        min(s_dims[1]+s_dims[2], l_dims[1]+l_dims[2]))

        fm_l_crop = l_map[:, \
                    round((overlap_dims[1] - l_dims[1]) * fm_scale_l): \
                    round((overlap_dims[3] - l_dims[1]) * fm_scale_l), \
                    round((overlap_dims[0] - l_dims[0]) * fm_scale_l): \
                    round((overlap_dims[2] - l_dims[0]) * fm_scale_l)] # B,fm_a_h,fm_a_w,c

        fm_s_crop = s_map[:, \
                    round((overlap_dims[1] - s_dims[1]) * fm_scale_s): \
                    round((overlap_dims[3] - s_dims[1]) * fm_scale_s), \
                    round((overlap_dims[0] - s_dims[0]) * fm_scale_s): \
                    round((overlap_dims[2] - s_dims[0]) * fm_scale_s)] # B,fm_b_h,fm_b_w,c
        
        fm_s_crop = F.interpolate(fm_s_crop.movedim(3,1), size=(fm_l_crop.shape[1], fm_l_crop.shape[2]), mode='bilinear', align_corners=False).movedim(1,3)

        if crop_dims[1][2] > crop_dims[0][2]:
            return fm_s_crop, fm_l_crop
        else:
            return fm_l_crop, fm_s_crop
