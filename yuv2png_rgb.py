import os, sys, argparse
import numpy as np
from PIL import Image

def yuv_to_rgb(y, u, v):
    # Convert YUV to RGB using standard formulas
    c = y - 16
    d = u - 128
    e = v - 128
    r = (298 * c + 409 * e + 128) >> 8
    g = (298 * c - 100 * d - 208 * e + 128) >> 8
    b = (298 * c + 516 * d + 128) >> 8

    # Clamp values to 0-255
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return r, g, b

def convert(yuv_file, output_dir):
    f_yuv = open(yuv_file, "rb")
    w, h = 3840, 2160
    seq_len = 129
    frame_size = int(3/2 * w * h)
    
    y_size = w * h
    uv_size = w * h // 4
    
    for frame_num in range(seq_len):
        converted_image = Image.new('RGB', (w, h))
        pixels = converted_image.load()
        
        f_yuv.seek(frame_size * frame_num)
        
        y_data = f_yuv.read(y_size)
        u_data = f_yuv.read(uv_size)
        v_data = f_yuv.read(uv_size)

        for i in range(h):
            for j in range(w):
                y = y_data[i * w + j]
                u = u_data[(i // 2) * (w // 2) + (j // 2)]
                v = v_data[(i // 2) * (w // 2) + (j // 2)]
                r, g, b = yuv_to_rgb(y, u, v)
                pixels[j, i] = (r, g, b)
        
        converted_image.save(os.path.join(output_dir, '%03d.png' % frame_num), "PNG")

    f_yuv.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yuv_file', type=str, required=True, help="Path to the YUV file")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Directory to save the output PNG images")
    args = parser.parse_args()

    yuv_file, output_dir = args.yuv_file, args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    convert(yuv_file, output_dir)