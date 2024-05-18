import os, sys, argparse
import numpy as np
from PIL import Image

def convert(yuv_file, output_dir):
    f_y = open(yuv_file, "rb")
    w ,h = 3840, 2160
    seq_len = 129
    frame_size = int(3/2 * w * h)
    for frame_num in range(seq_len):
        converted_image = Image.new('L', (w, h))
        pixels = converted_image.load()

        f_y.seek(frame_size * frame_num)
        
        for i in range(h):
            for j in range(w):
                y = ord(f_y.read(1))
                pixels[j,i] = int(y)

        converted_image.save(os.path.join(output_dir, '%03d.png' % frame_num), "png")

    f_y.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yuv_file', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    args = parser.parse_args()

    yuv_file, output_dir = args.yuv_file, args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    convert(yuv_file, output_dir)