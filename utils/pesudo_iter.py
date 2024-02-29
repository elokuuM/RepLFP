'''
Non-salient object supression make gt and mask
'''
import os
import cv2
import numpy as np
import json

sal_path = "../test_out/Swin-epo200-pretrained-0.001c/sp0"
mbp_path = "../test_out/mb200pred"
filled_correct_img_gt_path = '../dataset/train/filled_correct_img_gt'
filled_correct_mask_path = '../dataset/train/filled_correct_mask'
filled_img_pseudo_path = filled_correct_img_gt_path + '_mbud'
filled_mask_pseudo_path = filled_correct_mask_path + '_mbud'

# json_path = '/home/gaosy/DATA/Gao_DUTS_TR/json'
json_path = '../dataset/train/json'

if not os.path.exists(filled_img_pseudo_path):
    os.mkdir(filled_img_pseudo_path)

file_list = os.listdir(sal_path)
file_list.sort()

##------------------------------make gt--------------------------------##
for file in file_list:

    if (os.path.isfile(os.path.join(sal_path, file))):
        edge_map_orig = cv2.imread(os.path.join(sal_path, file), 0).astype(np.float32)
        mb_pred = cv2.imread(os.path.join(mbp_path, file), 0).astype(np.float32)
        intersection_map = (edge_map_orig/255.0) * (mb_pred/255.0) *255
        fore_gt = cv2.imread(os.path.join(filled_correct_img_gt_path, file.split('.')[0] + '.bmp'), 0).astype(np.float32)
        intersection_map[intersection_map < 200] = 0  # leave high-confidence foreground
        intersection_map[intersection_map == 255] = 254  # empty position 255
        # plt.imshow(edge_map_orig)
        # plt.show()
        edge_map = intersection_map.copy()

        data = json.load(open(os.path.join(json_path, file.split('.')[0] + '.json')))

        fore_ground_points = []
        for point in data['shapes']:
            if point['label'] == 'foreground':
                fore_ground_points.append(point['points'][0])

        for i, point in enumerate(fore_ground_points):
            print(point)
            seed_point = (int(point[0]), int(point[1]))
            print(i, ' : ', seed_point)

            if edge_map[seed_point[1], seed_point[0]] < 50:  # discard undetected salient points,
                continue

            # if edge_map[seed_point[1],seed_point[0]] < 50:  # use the initial pseudo-labels: employ edges and points to generate forground area
            #     pass

            mask = np.zeros([edge_map.shape[0] + 2, edge_map.shape[1] + 2], np.uint8)
            cv2.floodFill(edge_map, mask, seed_point, (255, 100, 100), (20, 20, 20), (50, 50, 50),
                          cv2.FLOODFILL_FIXED_RANGE)  # fill 255
            # print(edge_map.max())

        edge_map[
            edge_map != 255] = 0  # filter out non-saliet object, highlight salient object manually annotated by annotators

        foreground_map = edge_map + fore_gt
        foreground_map[foreground_map > 254] = 255
        cv2.imwrite(os.path.join(filled_img_pseudo_path, file), foreground_map)

##------------------------------------make mask-----------------------------##

if not os.path.exists(filled_mask_pseudo_path):
    os.mkdir(filled_mask_pseudo_path)

# generate the mask
for file in file_list:
    filled_background = cv2.imread(os.path.join(filled_correct_mask_path, file.split('.')[0] + '.bmp'), 0).astype(np.float32)
    filled_foreground = cv2.imread(os.path.join(filled_img_pseudo_path, file), 0).astype(np.float32)

    final_mask = filled_background + filled_foreground
    foreground_map[foreground_map > 254] = 255
    cv2.imwrite(os.path.join(filled_mask_pseudo_path, file), final_mask)

print('Finished NSS_2nd_GtMask')