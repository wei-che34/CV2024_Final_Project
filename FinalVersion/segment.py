import tensorflow as tf
import numpy as np
import cv2 as cv
from tensorflow.io import gfile
from PIL import Image
import tempfile
import os
from six.moves import urllib
import tarfile
import pandas as pd
from tqdm import tqdm

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None

        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image, INPUT_TENSOR_NAME = 'ImageTensor:0', OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.
            INPUT_TENSOR_NAME: The name of input tensor, default to ImageTensor.
            OUTPUT_TENSOR_NAME: The name of output tensor, default to SemanticPredictions.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        target_size = (2049,1025)  # size of Cityscapes images
        resized_image = image.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
        batch_seg_map = self.sess.run(
            OUTPUT_TENSOR_NAME,
            feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]  # expected batch size = 1
        if len(seg_map.shape) == 2:
            seg_map = np.expand_dims(seg_map,-1)  # need an extra dimension for cv.resize
        seg_map = cv.resize(seg_map, (width,height), interpolation=cv.INTER_NEAREST)
        return seg_map

MODEL_NAME = 'mobilenetv2_coco_cityscapes_trainfine'
#MODEL_NAME = 'xception65_cityscapes_trainfine'

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_cityscapes_trainfine':
        'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz',
    'xception65_cityscapes_trainfine':
        'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = tempfile.mkdtemp()
tf.io.gfile.makedirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')

def compute_M(seg_map, source_seg_map, source_img_path, target_img_path, sift, bf):
    '''
    seg_map: segmentation map of the source image, a numpy array of shape (height, width)
    source_img_path: path to the source image
    target_img_path: path to the target image
    sift: cv2.SIFT object
    bf: cv2.BFMatcher object
    '''

    unique_labels = np.unique(seg_map)

    source_img = cv.imread(source_img_path)
    target_img = cv.imread(target_img_path)

    gray_source = cv.cvtColor(source_img, cv.COLOR_BGR2GRAY)
    gray_target = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)

    # Function to extract and match features for each label
    def compute_transform(label, source_seg_map, seg_map, gray_frame_0, gray_frame_16):
        mask = seg_map == label
        source_mask = source_seg_map == label
        keypoints_0, descriptors_0 = sift.detectAndCompute(gray_frame_0, source_mask.astype(np.uint8))
        keypoints_16, descriptors_16 = sift.detectAndCompute(gray_frame_16, mask.astype(np.uint8))
            
        matches = bf.match(descriptors_0, descriptors_16)
        matches = sorted(matches, key=lambda x: x.distance)

        # only use the top 10% of matches
        top_matches = matches[:int(len(matches) * 0.1)]

        if len(top_matches) >= 4:  # Minimum number of matches to estimate a transform
            pts_0 = np.float32([keypoints_0[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
            pts_16 = np.float32([keypoints_16[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)
            # estimate projective transform
            M, mask = cv.estimateAffinePartial2D(pts_0, pts_16)
        elif len(matches) >= 4:
            print(f"Warning: Not enough matches for label {label}. Using all {len(matches)} matches.")
            pts_0 = np.float32([keypoints_0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_16 = np.float32([keypoints_16[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv.estimateAffinePartial2D(pts_0, pts_16)
        else:
            print(f"Not engough matches for label {label}. Using identical transform.")
            M = np.eye(2,3)
        
        return M
    
    # Compute transforms for each label
    transforms = {label: compute_transform(label, source_seg_map, seg_map, gray_source, gray_target) for label in unique_labels}

    return transforms
    
    
def global_motion_estimation(bf_path, af_path, target_path, bf_gray_path, af_gray_path, target_gray_path, model, curent_index):
    '''
    bf_path, af_path: RGB image for segmentation, a string
    bf_gray_path, af_gray_path, target_gray_path: gray image for feature extraction, a string
    model: DeepLabModel object
    '''

    # read teh gray images
    bf_gray = cv.imread(bf_gray_path)
    af_gray = cv.imread(af_gray_path)
    target_gray = cv.imread(target_gray_path)

    bf_gray = cv.cvtColor(bf_gray, cv.COLOR_BGR2GRAY)
    af_gray = cv.cvtColor(af_gray, cv.COLOR_BGR2GRAY)
    target_gray = cv.cvtColor(target_gray, cv.COLOR_BGR2GRAY)

    seg_map = model.run(Image.open(target_path))
    for y in range(16, target_gray.shape[0], 16):
        for x in range(16, target_gray.shape[1], 16):
            seperate = seg_map[y-16:y, x-16:x]
            seperate = seperate.reshape(256)
            counts = np.bincount(seperate)
            seg_map[y-16: y, x-16: x] = np.argmax(counts)

    # constrain the seg_map to have only 12 labels
    # combine the labels 2, 3, 14
    seg_map[seg_map == 3] = 2
    seg_map[seg_map == 14] = 2
    # combine the labels 1, 4
    seg_map[seg_map == 4] = 1
    # combine the labels 6, 7
    seg_map[seg_map == 7] = 6
    # combine the labes 11, 12
    seg_map[seg_map == 12] = 11
    # combine the labels 13, 15, 16
    seg_map[seg_map == 15] = 13
    seg_map[seg_map == 16] = 13
    # combine the labels 17, 18
    seg_map[seg_map == 18] = 17

    seg_map[seg_map == 5] = 3
    seg_map[seg_map == 6] = 4
    seg_map[seg_map == 8] = 5
    seg_map[seg_map == 10] = 6
    seg_map[seg_map == 11] = 7
    seg_map[seg_map == 13] = 8
    seg_map[seg_map == 17] = 9

    bf_seg_map = model.run(Image.open(bf_path))
    for y in range(16, target_gray.shape[0], 16):
        for x in range(16, target_gray.shape[1], 16):
            seperate = bf_seg_map[y-16:y, x-16:x]
            seperate = seperate.reshape(256)
            counts = np.bincount(seperate)
            bf_seg_map[y-16: y, x-16: x] = np.argmax(counts)

    # constrain the seg_map to have only 12 labels
    # combine the labels 2, 3, 14
    bf_seg_map[bf_seg_map == 3] = 2
    bf_seg_map[bf_seg_map == 14] = 2
    # combine the labels 1, 4
    bf_seg_map[bf_seg_map == 4] = 1
    # combine the labels 6, 7
    bf_seg_map[bf_seg_map == 7] = 6
    # combine the labes 11, 12
    bf_seg_map[bf_seg_map == 12] = 11
    # combine the labels 13, 15, 16
    bf_seg_map[bf_seg_map == 15] = 13
    bf_seg_map[bf_seg_map == 16] = 13
    # combine the labels 17, 18
    bf_seg_map[bf_seg_map == 18] = 17

    bf_seg_map[bf_seg_map == 5] = 3
    bf_seg_map[bf_seg_map == 6] = 4
    bf_seg_map[bf_seg_map == 8] = 5
    bf_seg_map[bf_seg_map == 10] = 6
    bf_seg_map[bf_seg_map == 11] = 7
    bf_seg_map[bf_seg_map == 13] = 8
    bf_seg_map[bf_seg_map == 17] = 9

    af_seg_map = model.run(Image.open(af_path))
    for y in range(16, target_gray.shape[0], 16):
        for x in range(16, target_gray.shape[1], 16):
            seperate = af_seg_map[y-16:y, x-16:x]
            seperate = seperate.reshape(256)
            counts = np.bincount(seperate)
            af_seg_map[y-16: y, x-16: x] = np.argmax(counts)

    # constrain the seg_map to have only 12 labels
    # combine the labels 2, 3, 14
    af_seg_map[af_seg_map == 3] = 2
    af_seg_map[af_seg_map == 14] = 2
    # combine the labels 1, 4
    af_seg_map[af_seg_map == 4] = 1
    # combine the labels 6, 7
    af_seg_map[af_seg_map == 7] = 6
    # combine the labes 11, 12
    af_seg_map[af_seg_map == 12] = 11
    # combine the labels 13, 15, 16
    af_seg_map[af_seg_map == 15] = 13
    af_seg_map[af_seg_map == 16] = 13
    # combine the labels 17, 18
    af_seg_map[af_seg_map == 18] = 17

    af_seg_map[af_seg_map == 5] = 3
    af_seg_map[af_seg_map == 6] = 4
    af_seg_map[af_seg_map == 8] = 5
    af_seg_map[af_seg_map == 10] = 6
    af_seg_map[af_seg_map == 11] = 7
    af_seg_map[af_seg_map == 13] = 8
    af_seg_map[af_seg_map == 17] = 9

    # store the map for each image to the disk
    f = open("./model_map/m_%03d.txt" % curent_index, 'w')
    for y in range(16, target_gray.shape[0], 16):
        for x in range(16, target_gray.shape[1], 16):
            if(y<1900):
                f.write(str(seg_map[y-16, x-16]) + '\n')
            else:
                f.write(str(10) + '\n')
    f.flush()
    f.close()

    # compute each M matrix for bf and af images' labels
    sift = cv.SIFT_create()
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    bf_M_dic = compute_M(seg_map, bf_seg_map, bf_gray_path, target_gray_path,   sift, bf)
    af_M_dic = compute_M(seg_map, af_seg_map, af_gray_path, target_gray_path,   sift, bf)

    # tansform
    bf_predict = np.zeros_like(target_gray)
    af_predict = np.zeros_like(target_gray)

    for label, M in bf_M_dic.items():
        print(M)
        mask = seg_map == label
        transformed_frame = cv.warpAffine(bf_gray, M, (bf_gray.shape[1], bf_gray.shape[0]))
        bf_predict[mask] = transformed_frame[mask]

    
    for label, M in af_M_dic.items():
        print(M)
        mask = seg_map == label
        transformed_frame = cv.warpAffine(af_gray, M, (af_gray.shape[1], af_gray.shape[0]))
        af_predict[mask] = transformed_frame[mask]

    # weighted sum
    alpha = 0.5
    target_predict = cv.addWeighted(bf_predict, alpha, af_predict, 1 - alpha, 0)

    for y in range(1900, target_predict.shape[0]):
        for x in range(target_predict.shape[1]):
            target_predict[y][x] = bf_gray[y][x]

    return target_predict

ref = pd.read_csv('reference.csv')
for i in tqdm(range(len(ref['target']))):
    print(f"start to process image{ref['target'][i]}")
    bf_path = './rgb_images/%03d.png' % ref['ref0'][i]
    af_path = './rgb_images/%03d.png' % ref['ref1'][i]
    target_path = './rgb_images/%03d.png' % ref['target'][i]

    bf_gray_path = './gt/%03d.png' % ref['ref0'][i]
    af_gray_path = './gt/%03d.png' % ref['ref1'][i]
    target_gray_path = './gt/%03d.png' % ref['target'][i]

    predict_frame = global_motion_estimation(bf_path, af_path, target_path, bf_gray_path, af_gray_path, target_gray_path, MODEL, int(ref['target'][i]))
    cv.imwrite('./solution/%03d.png' % ref['target'][i], predict_frame)
