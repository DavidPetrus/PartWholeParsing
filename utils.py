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
from sklearn.cluster import AgglomerativeClustering

from absl import flags

FLAGS = flags.FLAGS

#color = np.random.randint(0,256,[5120,3],dtype=np.uint8)

def random_crop(image):
    img_w, img_h = image.size

    crop_size = np.random.uniform(FLAGS.min_crop,FLAGS.max_crop)
    crop_size = int(min(img_h,img_w)*crop_size)
    crop_x,crop_y = np.random.randint(0,img_w-crop_size), np.random.randint(0,img_h-crop_size)

    crop = image.crop((crop_x, crop_y, crop_x+crop_size, crop_y+crop_size))
    #resized = F.interpolate(crop.unsqueeze(0),size=(FLAGS.image_size,FLAGS.image_size),mode='bilinear',align_corners=True).squeeze(0)

    return crop, [crop_x,crop_y,crop_size]

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
        agglom_clust = AgglomerativeClustering(n_clusters=None,distance_threshold=dist_thresh,affinity='cosine',linkage='average')
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
    agglom_clust = AgglomerativeClustering(n_clusters=None,distance_threshold=FLAGS.dist_thresh,affinity='cosine',linkage='average')
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