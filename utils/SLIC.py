import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries



img_root =  '../dataset/train/gray'
files = os.listdir(img_root)


for file in files:
    image = cv2.imread(os.path.join(img_root, file))
    #slic = cv2.ximgproc.createSuperpixelSLIC(img,region_size=20,ruler = 20.0) 
    #slic.iterate(10)     
    #mask_slic = slic.getLabelContourMask() 
    #label_slic = slic.getLabels()        
    #number_slic = slic.getNumberOfSuperpixels()  
    #mask_inv_slic = cv2.bitwise_not(mask_slic)  
    #img_slic = cv2.bitwise_and(img,img,mask =  mask_inv_slic) 
    segments_slic = slic(image, n_segments=200, compactness=3)
    img_slic = mark_boundaries(image, segments_slic)
    cv2.imshow("img_slic",img_slic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    