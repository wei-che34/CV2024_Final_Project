import os, sys, argparse
import numpy as np
from PIL import Image
import cv2


video=cv2.VideoWriter('video.yuv',-1,1,(3840,2160))
for i in range(129):
    if(i in [0, 32, 64, 96, 128]):
        video.write(cv2.imread('./gt/%03d.png' % i))
    else:
        video.write(cv2.imread('./solution/%03d.png' % i))

cv2.destroyAllWindows()
video.release()
