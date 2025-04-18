import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
#import matplotlib.pyplot as plt
import time
import pickle as pkl
import os
import json
from PIL import Image
import math

from absl import flags

FLAGS = flags.FLAGS

color = np.random.randint(0,256,[255,3],dtype=np.uint8)


def evaluate_accuracy(masks, labels):
    labels = F.one_hot(labels, FLAGS.num_output_classes).float()
    for b_ix in range(FLAGS.batch_size):



def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

def unnormalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.mul_(s)
        t.add_(m)
        
    return x

def transform_image(img):
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    
    return img

def random_crop(image, crop_dims=None, min_crop=0.5, inter_mode='bilinear'):
    c, img_h, img_w = image.shape

    crop_size = int(max(img_h,img_w)*crop_dims[2])
    crop_x, crop_y = int(crop_dims[0]*img_w), int(crop_dims[1]*img_h)
    crop = image[:, crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

    resized = F.interpolate(crop.unsqueeze(0),size=FLAGS.image_size,mode=inter_mode)

    return resized


def vic_reg(x):

    x = x.reshape(FLAGS.batch_size,-1,FLAGS.output_dim)
    x = x - x.mean(dim=1, keepdim=True)
    std_x = torch.sqrt(x.var(dim=1) + 0.0001)
    std_loss = torch.mean(F.relu(FLAGS.min_std - std_x))

    cov_x = torch.matmul(x.movedim(1,2),x) / (x.shape[1] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(FLAGS.batch_size*FLAGS.output_dim)

    return std_loss, cov_loss


def off_diagonal(x):
    b, n, m = x.shape
    assert n == m
    return x.reshape(b,-1)[:,:-1].reshape(b, n - 1, n + 1)[:,:,1:].reshape(b,-1)


def sinkhorn_knopp(sims):
    with torch.no_grad():
        Q = F.softmax(sims.reshape(-1) / FLAGS.epsilon, dim=0).reshape(-1, FLAGS.num_prototypes).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] # number of samples to assign
        K = Q.shape[0] # how many prototypes

        for it in range(FLAGS.sinkhorn_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q = Q/sum_of_rows
            Q = Q/K

            # normalize each column: total weight per sample must be 1/B
            Q = Q/torch.sum(Q, dim=0, keepdim=True)
            Q = Q/B

        if FLAGS.round_q:
            # Verify this is correct
            max_proto_sim,_ = Q.max(dim=0)
            Q[Q != max_proto_sim] = 0.
            Q[Q == max_proto_sim] = 1.
        else:
            Q = Q*B # the columns must sum to 1 so that Q is an assignment

        return Q.t()

def color_distortion(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2):
    color_jitter = torchvision.transforms.ColorJitter(brightness,contrast,saturation,hue)
    return color_jitter

def calculate_vars(log_dict, level_embds, pca):
    for l_ix, embd_tensor in enumerate(level_embds):
        embds = embd_tensor.detach().movedim(1,3).cpu().numpy()
        _,l_h,l_w,_ = embds.shape
        embds = embds.reshape(l_h*l_w,-1)

        fitted = pca.fit(embds)
        log_dict['var/comp1_l{}'.format(l_ix+1)] = fitted.explained_variance_[0]
        log_dict['var/comp2_l{}'.format(l_ix+1)] = fitted.explained_variance_[1]
        log_dict['var/comp3_l{}'.format(l_ix+1)] = fitted.explained_variance_[2]
        log_dict['var/comp4_l{}'.format(l_ix+1)] = fitted.explained_variance_[3:8].sum()
        log_dict['var/comp5_l{}'.format(l_ix+1)] = fitted.explained_variance_[8:].sum()

    return log_dict

def display_label(label):
    global color

    display = np.zeros([FLAGS.image_size, FLAGS.image_size, 3], dtype=np.uint8)
    for c in range(FLAGS.num_output_classes-1):
        display[label == c] = color[c]

    display[label == FLAGS.num_output_classes-1] = 0

    return display

def display_reconst_img(frame,reconst=None,segs=None,waitkey=False):
    if reconst is not None:
        imshow = reconst[0].detach().movedim(0,2).cpu().numpy() * 255.
        imshow = np.clip(imshow,0,255)
        imshow = imshow.astype(np.uint8)
        cv2.imshow('pred',imshow[:,:,::-1])

    targ = frame[0].detach().movedim(0,2).cpu().numpy() * 255.
    targ = targ.astype(np.uint8)
    cv2.imshow('target',targ[:,:,::-1])
    if segs is not None:
        for level,seg in enumerate(segs):
            cv2.imshow('L{}'.format(level+1),seg[:,:,::-1])

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key==27:
        exit()

def find_clusters(log_dict, level_embds):
    start = time.time()
    for dist_thresh in [0.1,0.2,0.3,0.5]:
        agglom_clust = AgglomerativeClustering(n_clusters=None,distance_threshold=dist_thresh,linkage='average')
        #for l_ix, embd_tensor in enumerate(level_embds):
        l_ix = 0
        embd_tensor = level_embds
        embds = embd_tensor.detach().movedim(1,3).cpu().numpy()
        _,l_h,l_w,_ = embds.shape
        embds = embds.reshape(l_h*l_w,-1)
        fitted = agglom_clust.fit(embds)
        log_dict['n_clusters/l{}_{}'.format(l_ix+1,dist_thresh)] = fitted.n_clusters_
    
    print('Clustering Time:',time.time()-start)
    return log_dict

def plot_embeddings(level_embds):
    global color

    resize = [8,8,8,8,16]
    segs = []
    agglom_clust = AgglomerativeClustering(n_clusters=None,distance_threshold=FLAGS.dist_thresh,linkage='average')
    for l_ix, embd_tensor in enumerate(level_embds):
        embds = embd_tensor.detach().movedim(1,3).cpu().numpy()
        _,l_h,l_w,_ = embds.shape
        embds = embds.reshape(l_h*l_w,-1)
        fitted = agglom_clust.fit(embds)
        #clusters = np.moveaxis(fitted.labels_.reshape(l_h,l_w),0,1)
        clusters = fitted.labels_.reshape(l_h,l_w)
        seg = np.zeros([clusters.shape[0],clusters.shape[1],3],dtype=np.uint8)
        for c in range(clusters.max()+1):
            seg[clusters==c] = color[c]
        seg = cv2.resize(seg, (seg.shape[1]*resize[l_ix],seg.shape[0]*resize[l_ix]))
        segs.append(seg)

    return segs


def loadAde20K(file):
    fileseg = file.replace('.jpg', '_seg.png');
    with Image.open(fileseg) as io:
        seg = np.array(io);

    # Obtain the segmentation mask, bult from the RGB channels of the _seg file
    R = seg[:,:,0];
    G = seg[:,:,1];
    B = seg[:,:,2];
    ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32));


    # Obtain the instance mask from the blue channel of the _seg file
    Minstances_hat = np.unique(B, return_inverse=True)[1]
    Minstances_hat = np.reshape(Minstances_hat, B.shape)
    ObjectInstanceMasks = Minstances_hat


    level = 0
    PartsClassMasks = [];
    PartsInstanceMasks = [];
    while True:
        level = level+1;
        file_parts = file.replace('.jpg', '_parts_{}.png'.format(level));
        if os.path.isfile(file_parts):
            with Image.open(file_parts) as io:
                partsseg = np.array(io);
            R = partsseg[:,:,0];
            G = partsseg[:,:,1];
            B = partsseg[:,:,2];
            PartsClassMasks.append((np.int32(R)/10)*256+np.int32(G));
            PartsInstanceMasks = PartsClassMasks
            # TODO:  correct partinstancemasks

            
        else:
            break

    objects = {}
    parts = {}

    attr_file_name = file.replace('.jpg', '.json')
    if os.path.isfile(attr_file_name):
        with open(attr_file_name, 'r') as f:
            input_info = json.load(f)

        contents = input_info['annotation']['object']
        instance = np.array([int(x['id']) for x in contents])
        names = [x['raw_name'] for x in contents]
        corrected_raw_name =  [x['name'] for x in contents]
        partlevel = np.array([int(x['parts']['part_level']) for x in contents])
        ispart = np.array([p>0 for p in partlevel])
        iscrop = np.array([int(x['crop']) for x in contents])
        listattributes = [x['attributes'] for x in contents]
        polygon = [x['polygon'] for x in contents]
        for p in polygon:
            p['x'] = np.array(p['x'])
            p['y'] = np.array(p['y'])

        objects['instancendx'] = instance[ispart == 0]
        objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
        objects['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 0)[0])]
        objects['iscrop'] = iscrop[ispart == 0]
        objects['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 0)[0])]
        objects['polygon'] = [polygon[x] for x in list(np.where(ispart == 0)[0])]


        parts['instancendx'] = instance[ispart == 1]
        parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
        parts['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 1)[0])]
        parts['iscrop'] = iscrop[ispart == 1]
        parts['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 1)[0])]
        parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

    return {'img_name': file, 'segm_name': fileseg,
            'class_mask': ObjectClassMasks, 'instance_mask': ObjectInstanceMasks, 
            'partclass_mask': PartsClassMasks, 'part_instance_mask': PartsInstanceMasks, 
            'objects': objects, 'parts': parts}


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)