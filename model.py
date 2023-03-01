import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from sklearn.cluster import AgglomerativeClustering
from utils import find_clusters

from absl import flags

FLAGS = flags.FLAGS

class ImageParser(nn.Module):

    def __init__(self, clip_model, objectnames):
        super(ImageParser, self).__init__()

        self.clip_model = clip_model
        '''proj_layers = []
        for l in range(FLAGS.proj_depth-1):
            proj_layers.append(nn.Conv2d(640, 640, kernel_size=1))
            proj_layers.append(nn.ReLU())
        proj_layers.append(nn.Conv2d(640, FLAGS.proj_dim, kernel_size=1))
        self.proj_wholes = nn.Sequential(*proj_layers)'''

        with torch.no_grad():
            self.get_text_embeddings(objectnames)
        self.fm_dim = 24

    def get_text_embeddings(self, objectnames):
        prototypes = []
        for category in objectnames:
            text = clip.tokenize([f"a {category}"]).to('cuda')
            prototypes.append(self.clip_model.encode_text(text))

        self.prototypes = nn.Parameter(torch.cat(prototypes,dim=0), requires_grad=True).to('cuda')

    def get_feature_maps(self, x):
        with torch.set_grad_enabled(FLAGS.train_clip):
            if FLAGS.clip_model == 'vit':
                x = self.clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
                # shape = [*, grid ** 2 + 1, width]
                x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
                x = self.clip_model.visual.ln_pre(x)

                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.clip_model.visual.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                
                feature_map = x.clone()

                #x = self.clip_model.visual.ln_post(x[:, 0, :])
                x = self.clip_model.visual.ln_post(x)

                x = x @ self.clip_model.visual.proj

                return x[:, 1:].reshape(FLAGS.batch_size, 24, 24, 768).detach()
            else:
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

        return self.proj_wholes(x.float()).movedim(1,3)

    def compute_sim_loss(self, x, class_map):
        norm_protos = F.normalize(self.prototypes, dim=1)
        sims = F.normalize(x, dim=3)[:,:,:,None] @ norm_protos[class_map-1][:,:,:,:,None] # BS, self.fm_dim, self.fm_dim

        return 1-sims.mean()

    def get_acc(self, x, class_map):
        sims = (F.normalize(x, dim=3) @ F.normalize(self.prototypes, dim=1).T) # BS, self.fm_dim, self.fm_dim, num_prototypes

        acc = (torch.argmax(sims, dim=3) == class_map-1).float().mean()

        return acc


    def vic_reg(self, x):
        std_x = torch.sqrt(x.var(dim=1) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x))

        x = x.reshape(FLAGS.batch_size*self.fm_dim*self.fm_dim, FLAGS.proj_dim)
        x = x - x.mean(dim=0, keepdim=True)
        cov_x = torch.matmul(x.movedim(0,1),x) / (x.shape[0] - 1)
        cov_loss = (self.off_diagonal(cov_x)).pow_(2).sum().div(x.shape[0]*x.shape[1])

        return std_loss, cov_loss, std_x.mean()


    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.reshape(-1)[:-1].reshape(n - 1, n + 1)[:,1:].reshape(-1)




