'''
Non-salient object supression make gt and mask
'''
import os
import cv2
import numpy as np

pesudo_path = "../test_out/1round_CRF"
unsupervise_path = '../test_out/MB_CRF'
mixed_pesudo_path = '../dataset/train/mixed_gt'



if not os.path.exists(mixed_pesudo_path):
    os.mkdir(mixed_pesudo_path)

file_list = os.listdir(pesudo_path)
file_list.sort()
a = []
##------------------------------mixed gt--------------------------------##
for file in file_list:

    if (os.path.isfile(os.path.join(pesudo_path, file))):
        pesudo_img = cv2.imread(os.path.join(pesudo_path, file), 0).astype(np.float32)
        pesudo_img = cv2.normalize(pesudo_img, None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        unsup_img  = cv2.imread(os.path.join(unsupervise_path, file), 0).astype(np.float32)
        unsup_img  = cv2.normalize(unsup_img, None, 0.0, 1.0, cv2.NORM_MINMAX)
        uw_mea = np.mean(np.abs(pesudo_img - unsup_img))
        a.append({file: uw_mea})
        if(uw_mea<0.1):
            mixed_pesudo = ((pesudo_img+unsup_img)/2 * 255).astype(np.uint8)
        else:
            mixed_pesudo = (pesudo_img* 255).astype(np.uint8)
            
        cv2.imwrite(os.path.join(mixed_pesudo_path, file), mixed_pesudo)


print(a)
print('Finished PesudomixMb')