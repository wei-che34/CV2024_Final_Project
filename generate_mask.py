import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import json

def show_anns(anns, i):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
    
path = '../rgb_images'

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


for i in tqdm(range(129)):
    if i<10:
        image = cv2.imread(os.path.join(path, '00'+str(i)+'.png'))
    elif i<100:
        image = cv2.imread(os.path.join(path, '0'+str(i)+'.png'))
    else:
        image = cv2.imread(os.path.join(path, str(i)+'.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sys.path.append("..")

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32, box_nms_thresh = 0.7)
    masks = mask_generator.generate(image)
    masks = [mask for mask in masks if mask['area'] > 100000]
    # plt.figure(figsize=(20,20))
    plt.imshow(image, alpha=0)
    show_anns(masks, i)
    plt.axis('off')
    #save the image with masks
    plt.savefig('../masks_images/'+str(i)+'.png')
    plt.clf()
    masks_dict = {}
    for j in range(len(masks)):
        mask_dict = {}
        mask_dict['segmentation'] = masks[j]['segmentation'].tolist()
        mask_dict['bbox'] = masks[j]['bbox']
        masks_dict[j] = mask_dict
    #save the masks in json format
    with open('../masks_json/'+str(i)+'.json', 'w') as f:
        json.dump(masks_dict, f)