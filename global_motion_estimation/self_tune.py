import motion as motion
import numpy as np
import cv2
import os

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

previous_path = '/Users/caichengyun/Desktop/CV/FinalProject/png/000.png'
current_path = '/Users/caichengyun/Desktop/CV/FinalProject/png/016.png'
future_path = '/Users/caichengyun/Desktop/CV/FinalProject/png/032.png'


img_0 = cv2.imread(previous_path, cv2.COLOR_BGR2GRAY)
img_16 = cv2.imread(current_path, cv2.COLOR_BGR2GRAY)
img_32 = cv2.imread(future_path, cv2.COLOR_BGR2GRAY)

params = motion.global_motion_estimation(img_0, img_16)

model_motion_field = motion.get_motion_field_affine(
    (int(img_0.shape[0] / motion.BBME_BLOCK_SIZE), int(img_0.shape[1] / motion.BBME_BLOCK_SIZE), 2), parameters=params
)

compensated_0 = motion.compensate_frame(img_0, model_motion_field)

params = motion.global_motion_estimation(img_32, img_16)

model_motion_field = motion.get_motion_field_affine(
    (int(img_32.shape[0] / motion.BBME_BLOCK_SIZE), int(img_32.shape[1] / motion.BBME_BLOCK_SIZE), 2), parameters=params
)

compensated_32 = motion.compensate_frame(img_32, model_motion_field)

compensated = cv2.addWeighted(compensated_0, 0.5, compensated_32, 0.5, 0)

cv2.imwrite("compensated.png", compensated)

psnr = PSNR(img_16, compensated)

# compute l2 distance
l2 = np.linalg.norm(img_0 - compensated)

print(f'PSNR: {psnr}')
print(f'L2 distance: {l2}')