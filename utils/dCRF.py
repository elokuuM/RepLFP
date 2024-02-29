import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import PIL.Image as Image
import multiprocessing
# from evaluate_sal import fm_and_mae
from skimage.segmentation import slic, watershed, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.filters import sobel
from skimage.color import rgb2gray
from tqdm import tqdm
import cv2





img_root =  '../dataset/train/gray'
cam_root = "../test_out/1round-epo200-0.001clsc-1_6dali/sp0"
output_root = "../test_out/1round_CRF/"
#cam_root = '../dataset/train/mixed_gt'
#output_root = "../test_out/mixed_gt_crf/"
#cam_root = "../dataset/train/MB+"
#output_root = "../test_out/MB_CRF/"



if not os.path.exists(output_root):
    os.mkdir(output_root)

cam_files = []
cam_walk = os.walk(cam_root)

for root, _, files in cam_walk:
    for file in files:
        cam_files.append(os.path.join(root, file).split('/')[-2:])



def camTcrf(image_dir):
    img = Image.open(os.path.join(img_root , image_dir[1][:-4] + '.bmp'))

    W, H = img.size
    img = np.array(img, dtype=np.uint8)
    img = cv2.equalizeHist(img)
    cam = Image.open(os.path.join(cam_root , image_dir[1][:-4] + '.png'))
    cam = cam.resize((W, H))
    probs = np.array(cam)

    probs = probs.astype(np.float64)/255.0
    probs[probs>0.5] = 1

    probs = np.concatenate((1 - probs[None, ...], probs[None, ...]), 0)
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2) #2：n_labels 背景和前景

    # get unary potentials (neg log probability)
    U = unary_from_softmax(probs) #从softmax概率计算一元势：U = -np.log(py)  probs:2xhxw
    d.setUnaryEnergy(U)

    # This creates the color-dependent features and then add them to the CRF
    # feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
    #                                   img=img, chdim=2) #颜色特征
    # d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC) #颜色无关特征，即位置特征
    # d.addPairwiseEnergy(feats, compat=10,
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
    pairwise_energy = create_pairwise_bilateral(sdims=(60, 60), schan=(5,), img=img, chdim=-1) #chdim 图像通道所在维度
    d.addPairwiseEnergy(pairwise_energy, compat=5)
    # Run five inference steps.
    Q = d.inference(10)

    MAP = np.array(Q)[1].reshape((H, W)) #取前景
    MAP = (MAP*255).astype(np.uint8)
    msk = Image.fromarray(MAP)
    msk.save(os.path.join(output_root, image_dir[1]), 'png')




if __name__ == '__main__':
    for file in tqdm(cam_files):
        camTcrf(file)

