import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torchvision.models.feature_extraction import create_feature_extractor

from utils import find_clusters

from absl import flags

FLAGS = flags.FLAGS

class ImageParser(nn.Module):

    def __init__(self, arch):
        super(ImageParser, self).__init__()

        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to('cuda')
        dino_resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        return_nodes = {
            'layer1.2.add': 'layer1',
            'layer2.3.add': 'layer2',
        }
        self.dino_resnet = create_feature_extractor(dino_resnet, return_nodes=return_nodes).to('cuda')

        self.conv_x14 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding='same'), nn.BatchNorm2d(384), nn.ReLU(), nn.Upsample(scale_factor=1.75, mode='bilinear'),
            nn.Conv2d(384, 384, kernel_size=3, padding='same'), nn.BatchNorm2d(384), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(384, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(384)
        ).to('cuda')
        self.conv_x8 = nn.Sequential(
            nn.Conv2d(512, 384, kernel_size=3, padding='same'), nn.BatchNorm2d(384), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding='same'), nn.BatchNorm2d(384), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(384, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(384)
        ).to('cuda')
        self.conv_x4 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding='same'), nn.BatchNorm2d(384), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding='same'), nn.BatchNorm2d(384), nn.ReLU(),
            nn.Conv2d(384, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(384)
        ).to('cuda')

        self.proj_layer = nn.Conv2d(FLAGS.embd_dim, FLAGS.output_dim, kernel_size=1).to('cuda')
        self.clusters = nn.Parameter(torch.randn(FLAGS.num_output_classes, FLAGS.embd_dim)).to('cuda')
        self.dino_clusters = nn.Parameter(torch.randn(FLAGS.num_output_classes, 384)).to('cuda')

    def cluster_lookup(self, x, dino_cluster=False):
        normed_clusters = F.normalize(self.dino_clusters, dim=1) if dino_cluster else F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)

        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), FLAGS.num_output_classes).permute(0, 3, 1, 2).to(torch.float32)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        return cluster_loss, cluster_probs
    

    def forward(self, x, val=False):
        bs = FLAGS.miou_bs if val else FLAGS.batch_size
        v2_fm_size = FLAGS.eval_size//14 if val else FLAGS.image_size//14
        with torch.set_grad_enabled(FLAGS.train_dinov2):
            # get dino activations
            res = self.dinov2(x, is_training=True)
            feats_s14 = res['x_prenorm'][:,1:].reshape(bs, v2_fm_size, v2_fm_size, 384).movedim(3,1) # bs,c,h,w

        with torch.set_grad_enabled(FLAGS.train_dino_resnet):
            ret_dict = self.dino_resnet(x)
            feats_s4 = ret_dict['layer1']
            feats_s8 = ret_dict['layer2']

        out_features = self.conv_x14(feats_s14) + self.conv_x8(feats_s8) + self.conv_x4(feats_s4)
        masks = self.proj_layer(out_features).movedim(1,3) # bs,h,w,c

        return masks, out_features, feats_s14

    def match_crops(self, sims_a, sims_b, crop_dims):
        #sims_a = sims_a.reshape(FLAGS.batch_size, self.fm_size, self.fm_size, FLAGS.output_dim)
        #sims_b = sims_b.reshape(FLAGS.batch_size, self.fm_size, self.fm_size, FLAGS.output_dim)
        b,fm_size,_,c = sims_a.shape
        if crop_dims[0][3] == True:
            sims_a = torchvision.transforms.functional.hflip(sims_a.movedim(3,1)).movedim(1,3)
        if crop_dims[1][3] == True:
            sims_b = torchvision.transforms.functional.hflip(sims_b.movedim(3,1)).movedim(1,3)

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
