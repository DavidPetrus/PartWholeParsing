import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from absl import flags

FLAGS = flags.FLAGS

class ImageParser(nn.Module):

    def __init__(self, clip_model):
        super(ImageParser, self).__init__()

        self.clip_model = clip_model
        proj_layers = []
        for l in range(FLAGS.proj_depth-1):
            proj_layers.append(nn.Conv2d(640, 640, kernel_size=1))
            proj_layers.append(nn.ReLU())
        proj_layers.append(nn.Conv2d(640, FLAGS.proj_dim, kernel_size=1))
        self.proj_wholes = nn.Sequential(*proj_layers)

    def get_feature_maps(self, x):
        with torch.set_grad_enabled(FLAGS.train_clip):
            def stem(x):
                x = self.clip_model.visual.relu1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
                x = self.clip_model.visual.relu2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
                x = self.clip_model.visual.relu3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
                x = self.clip_model.visual.avgpool(x)
                return x

            x = x.type(self.clip_model.visual.conv1.weight.dtype)
            x = stem(x)
            x = self.clip_model.visual.layer1(x)
            x = self.clip_model.visual.layer2(x)
            #x = self.visual.layer3(x)
            #x = self.visual.layer4(x)
            #x = self.visual.attnpool(x)

        if not FLAGS.train_clip:
            x = x.detach()

        return self.proj_wholes(x.float())

    def compute_dists(self, x, class_map):
        class_map = class_map.reshape(FLAGS.batch_size,36*36)
        uniq,inv_idxs = torch.unique(class_map, return_inverse=True)
        class_mask = F.one_hot(inv_idxs).reshape(FLAGS.batch_size, 1, 36*36, len(uniq))

        x = x.reshape(FLAGS.batch_size, FLAGS.proj_dim, 36*36, 1)
        mean_embds = (x * class_mask).sum(dim=2,keepdim=True) / (class_mask.sum(dim=2,keepdim=True) + 0.0001) # BS, proj_dim, 1, num_classes
        dists = ((x - (mean_embds*class_mask).sum(dim=3,keepdim=True))**2).mean(dim=1)

        return dists

    def vic_reg(self, x):
        std_x = torch.sqrt(x.var(dim=1) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x))

        x = x.reshape(FLAGS.batch_size*36*36, FLAGS.proj_dim)
        x = x - x.mean(dim=0, keepdim=True)
        cov_x = torch.matmul(x.movedim(0,1),x) / (x.shape[0] - 1)
        cov_loss = (self.off_diagonal(cov_x)).pow_(2).sum().div(x.shape[0]*x.shape[1])

        return std_loss, cov_loss, std_x.mean()


    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.reshape(-1)[:-1].reshape(n - 1, n + 1)[:,1:].reshape(-1)





