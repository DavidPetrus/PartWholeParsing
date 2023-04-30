import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torchvision.models.feature_extraction import create_feature_extractor
import dino.vision_transformer as vits

from utils import find_clusters

from absl import flags

FLAGS = flags.FLAGS

class ImageParser(nn.Module):

    def __init__(self, arch, dropout=0.):
        super(ImageParser, self).__init__()

        if FLAGS.backbone == 'dinov1':
            self.dinov1 = vits.__dict__[arch](
                patch_size=8,
                num_classes=0)
            if not FLAGS.train_dinov1:
                for p in self.dinov1.parameters():
                    p.requires_grad = False
            self.dinov1.eval().cuda()

            if arch == "vit_small":
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            elif arch == "vit_base":
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.dinov1.load_state_dict(state_dict, strict=True)
        elif FLAGS.backbone == 'dinov2':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to('cuda')
            dino_resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            return_nodes = {
                'layer1.2.add': 'layer1',
                'layer2.3.add': 'layer2',
            }
            self.dino_resnet = create_feature_extractor(dino_resnet, return_nodes=return_nodes).to('cuda')

        if FLAGS.seg_layers == 'conv':
            convs = []
            for l in range(FLAGS.depth):
                convs.extend([nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim), nn.GELU()])
            self.conv_x14 = nn.Sequential(
                *convs,
                torch.nn.Dropout(p=dropout), nn.Upsample(scale_factor=1.75, mode='bilinear'),
                nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim), nn.GELU(), nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim)
            ).to('cuda')
            convs = []
            for l in range(FLAGS.depth):
                convs.extend([nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim), nn.GELU()])
            self.conv_x8 = nn.Sequential(
                nn.Conv2d(512, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim), nn.GELU(),
                *convs,
                torch.nn.Dropout(p=dropout), nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim)
            ).to('cuda')
            convs = []
            for l in range(FLAGS.depth):
                convs.extend([nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim), nn.GELU()])
            self.conv_x4 = nn.Sequential(
                nn.Conv2d(256, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim), nn.GELU(),
                *convs,
                torch.nn.Dropout(p=dropout), nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=3, padding='same'), nn.BatchNorm2d(FLAGS.embd_dim)
            ).to('cuda')
        elif FLAGS.seg_layers == 'attn':
            attn_layers = []
            for l in range(FLAGS.depth):
                attn_layers.append(AttnBlock(
                    dim=FLAGS.embd_dim,
                    num_heads=6,
                    mlp_ratio=2,
                    drop=0,
                    attn_drop=0,
                    drop_path=0)
                )
                
            attn_layers.append(nn.Linear(FLAGS.embd_dim, FLAGS.embd_dim))

            self.attn_layers = nn.Sequential(*attn_layers).to('cuda')

        proj_mlp = []
        for l in range(FLAGS.proj_depth):
            proj_mlp.extend([nn.Conv2d(FLAGS.embd_dim, FLAGS.embd_dim, kernel_size=1), nn.GELU()])
        self.proj_mlp = nn.Sequential(*proj_mlp).to('cuda')
        self.proj_layer = torch.nn.utils.weight_norm(nn.Conv2d(FLAGS.embd_dim, FLAGS.output_dim, kernel_size=1)).to('cuda')
        self.proj_layer.weight_g.data.fill_(1)
        self.proj_layer.weight_g.requires_grad = False
    

    def forward(self, x, val=False, student=False):
        bs = FLAGS.miou_bs if val else FLAGS.batch_size
        if FLAGS.backbone == 'dinov1':
            v1_fm_size = FLAGS.eval_size//8 if val else FLAGS.image_size//8
            with torch.set_grad_enabled(FLAGS.train_dinov1):
                # get dino activations
                feats_s8 = self.dinov1(x) # bs, h*w+1, c
                if student:
                    rand_mask = (torch.rand(feats_s8.shape[0], feats_s8.shape[1], 1, device='cuda') > FLAGS.patch_masking).float()
                    feats_s8 = feats_s8 * rand_mask

        elif FLAGS.backbone == 'dinov2':
            v2_fm_size = FLAGS.eval_size//14 if val else FLAGS.image_size//14
            with torch.set_grad_enabled(FLAGS.train_dinov2):
                # get dino activations
                res = self.dinov2(x, is_training=True)
                feats_s14 = res['x_prenorm'][:,1:].reshape(bs, v2_fm_size, v2_fm_size, 384).movedim(3,1) # bs,c,h,w

            with torch.set_grad_enabled(FLAGS.train_dino_resnet):
                ret_dict = self.dino_resnet(x)
                feats_s4 = ret_dict['layer1']
                feats_s8 = ret_dict['layer2']

        if FLAGS.seg_layers == 'conv':
            out_features = self.conv_x14(feats_s14) + self.conv_x8(feats_s8) + self.conv_x4(feats_s4) # bs,c,h,w
        elif FLAGS.seg_layers == 'attn':
            out_features = self.attn_layers(feats_s8)
            out_features = out_features[:,1:].reshape(bs, v1_fm_size, v1_fm_size, FLAGS.embd_dim).movedim(3,1) # bs,c,h,w
            out_features = F.upsample(out_features, scale_factor=2)
        
        masks = self.proj_layer(F.normalize(self.proj_mlp(out_features), dim=1)) # bs,c,h,w
        if student and FLAGS.fm_noise > 0.:
            fm_noise = torch.randn(*masks.shape, device='cuda') * FLAGS.fm_noise
            masks = masks + fm_noise

        return masks

    def match_crops(self, sims_a, sims_b, crop_dims):
        #sims_a = sims_a.reshape(FLAGS.batch_size, FLAGS.output_dim, self.fm_size, self.fm_size)
        #sims_b = sims_b.reshape(FLAGS.batch_size, FLAGS.output_dim, self.fm_size, self.fm_size)
        b,c,fm_size,_ = sims_a.shape
        if crop_dims[0][3] == True:
            sims_a = torchvision.transforms.functional.hflip(sims_a)
        if crop_dims[1][3] == True:
            sims_b = torchvision.transforms.functional.hflip(sims_b)

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

        fm_l_crop = l_map[:,:, \
                    round((overlap_dims[1] - l_dims[1]) * fm_scale_l): \
                    round((overlap_dims[3] - l_dims[1]) * fm_scale_l), \
                    round((overlap_dims[0] - l_dims[0]) * fm_scale_l): \
                    round((overlap_dims[2] - l_dims[0]) * fm_scale_l)] # B,c,fm_a_h,fm_a_w

        fm_s_crop = s_map[:,:, \
                    round((overlap_dims[1] - s_dims[1]) * fm_scale_s): \
                    round((overlap_dims[3] - s_dims[1]) * fm_scale_s), \
                    round((overlap_dims[0] - s_dims[0]) * fm_scale_s): \
                    round((overlap_dims[2] - s_dims[0]) * fm_scale_s)] # B,c,fm_b_h,fm_b_w
        
        fm_s_crop = F.interpolate(fm_s_crop, size=(fm_l_crop.shape[2], fm_l_crop.shape[3]), mode='bilinear', align_corners=False)

        if crop_dims[1][2] > crop_dims[0][2]:
            return fm_s_crop, fm_l_crop
        else:
            return fm_l_crop, fm_s_crop


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        assert key is None
        assert value is None
        x = query
        B, N, C = x.shape
        S = N
        # [3, B, nh, N, C//nh]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, nh, N, C//nh]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=FLAGS.dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=FLAGS.dropout)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x