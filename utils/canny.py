import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries



img_root =  '../dataset/train/gray'
files = np.sort(os.listdir(img_root))


for file in files:
    image = cv2.imread(os.path.join(img_root, file))
    #slic = cv2.ximgproc.createSuperpixelSLIC(img,region_size=20,ruler = 20.0) 
    #slic.iterate(10)     
    #mask_slic = slic.getLabelContourMask() 
    #label_slic = slic.getLabels()        
    #number_slic = slic.getNumberOfSuperpixels()  
    #mask_inv_slic = cv2.bitwise_not(mask_slic)  
    #img_slic = cv2.bitwise_and(img,img,mask =  mask_inv_slic) 
    edges = cv2.Canny(image,20,100)
    cv2.imshow(file,edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    