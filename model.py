import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import dino.vision_transformer as vits
from utils import find_clusters

from absl import flags

FLAGS = flags.FLAGS

class ImageParser(nn.Module):

    def __init__(self, arch):
        super(ImageParser, self).__init__()

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

        if FLAGS.feed_labels:
            self.lab_net = nn.Linear(FLAGS.num_output_classes+FLAGS.embd_dim, FLAGS.embd_dim).to('cuda')

        seg_layers = []
        if not FLAGS.mlp_only:
            for l in range(FLAGS.depth):
                seg_layers.append(AttnBlock(
                        dim=FLAGS.embd_dim,
                        num_heads=6,
                        mlp_ratio=2,
                        drop=0,
                        attn_drop=0,
                        drop_path=0))

        else:
            seg_layers.append(nn.Linear(FLAGS.embd_dim, FLAGS.embd_dim))
            seg_layers.append(nn.ReLU())
            
        seg_layers.append(nn.Linear(FLAGS.embd_dim, FLAGS.embd_dim))

        self.seg_layers = nn.Sequential(*seg_layers).to('cuda')

        self.class_pred = nn.Linear(FLAGS.embd_dim, FLAGS.num_output_classes).to('cuda')

        self.proj_head = vits.DINOHead(FLAGS.embd_dim, FLAGS.output_dim).to('cuda')

    def forward(self, x, labels=None):
        self.model.eval()
        with torch.no_grad():
            # get dino activations
            dino_feat = self.model(x)

        if FLAGS.feed_labels:
            labels = F.interpolate(labels.unsqueeze(1).float(), size=28, mode='nearest').reshape(FLAGS.batch_size,-1).long()
            dino_feat = self.lab_net(torch.cat([dino_feat[:,1:], F.one_hot(labels,FLAGS.num_output_classes).float()], dim=-1))
            feat = self.seg_layers(dino_feat)
        else:
            feat = self.seg_layers(dino_feat)[:,1:]

        proj_feat = self.proj_head(feat)

        return feat, proj_feat

    def match_crops(self, sims_a, sims_b, crop_dims):
        sims_a = sims_a.reshape(FLAGS.batch_size, FLAGS.image_size//8, FLAGS.image_size//8, -1)
        sims_b = sims_b.reshape(FLAGS.batch_size, FLAGS.image_size//8, FLAGS.image_size//8, -1)
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
                 qkv_fuse=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3)
        else:
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

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
            proj_drop=drop,
            qkv_fuse=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
