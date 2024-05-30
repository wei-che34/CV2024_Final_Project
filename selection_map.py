import os, sys, argparse
import numpy as np
from PIL import Image
import heapq

def cal_psnr(pred_img, gt_img):

    psnr = []
    s = np.array(Image.open(pred_img).convert('L'))
    g = np.array(Image.open(gt_img).convert('L'))
        
    s = s.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16)
    g = g.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16)

    for i in range(s.shape[0]):
    #print(s.shape)
        mse = np.sum((s[i]-g[i])**2)
        #/s.size
        psnr.append(10*np.log10(255**2/mse))

    #psnr = np.array(psnr)

    return psnr

img_num = [i for i in range(129) if i not in [0, 32, 64, 96, 128]]
for i in img_num:
    pred_img = "./solution/%03d.png" % i
    gt_img = "./gt/%03d.png" % i
    psnr = cal_psnr(pred_img,gt_img)
    max_number = heapq.nlargest(13000, psnr) 
    map = []
    for t in max_number:
        index = psnr.index(t)
        map.append(index)
        psnr[index] = -10000
    f = open("./solution/s_%03d.txt" % i, 'w')
    for k in range(len(psnr)):
        if(k in map):
            f.write("1\n")
        else:
            f.write("0\n")
    f.flush()
    f.close