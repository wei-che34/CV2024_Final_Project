import os, sys, argparse
import numpy as np
from PIL import Image

def benchmark(so_path, gt_path):

    image_name = ['%03d.png'% i for i in range(129) if i not in [0, 32, 64, 96, 128]]
    txt_name   = ['s_%03d.txt'% i for i in range(129) if i not in [0, 32, 64, 96, 128]]

    so_img_paths = [os.path.join(so_path,name) for name in image_name]
    so_txt_paths = [os.path.join(so_path,name) for name in txt_name]
    gt_img_paths = [os.path.join(gt_path,name) for name in image_name]

    psnr = []
    for so_img_path, so_txt_path, gt_img_path in zip(so_img_paths, so_txt_paths, gt_img_paths):
        
        print('check image... ', so_img_path)

        s = np.array(Image.open(so_img_path).convert('L'))
        g = np.array(Image.open(gt_img_path).convert('L'))
        f = open(so_txt_path, 'r')

        mask = []
        for line in f.readlines():
            mask.append(int(line.strip('\n')))
        f.close()
        
        mask = np.array(mask).astype(bool)
        assert np.sum(mask) == 13000, 'The number of selection blocks should be 13000'


        s = s.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16)
        g = g.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16)
        
        s = s[mask]
        g = g[mask]
        assert not (s == g).all(), "The prediction should not be the same as the ground truth"

        mse = np.sum((s-g)**2)/s.size
        psnr.append(10*np.log10(255/mse))

    
    psnr = np.array(psnr)
    avg_psnr = np.sum(psnr) / len(psnr)


    return avg_psnr




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--so_path', type=str)
    parser.add_argument('-g', '--gt_path', type=str)
    args = parser.parse_args()

    so_path = args.so_path
    gt_path = args.gt_path
     
    score = benchmark(so_path, gt_path)

    print('PSNR: %.5f\n'%(score))