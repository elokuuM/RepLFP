import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import cv2
import torch
import os
from torchvision import transforms

from scipy.ndimage.interpolation import rotate
import torch.nn.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt):
        # assert img.size == mask.size
        if img.size == gt.size:
            pass
        else:
            print(img.size, gt.size)

        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt


class RandomHorizontallyFlip(object):
    def __call__(self, img, gt):
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT) 
        return img, gt

class RandomVerticallyFlip(object):
    def __call__(self, img, gt):
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), gt.transpose(Image.FLIP_TOP_BOTTOM) 
        return img, gt

class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("size参数请设置为int或者tuple")

    def __call__(self, img, mask):
        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)
        return img, mask


class RandomRotate(object):
    def __call__(self, img, mask, edge, angle_range=(0, 180)):
        self.degree = np.random.randint(*angle_range)
        rotate_degree = np.random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST), edge.rotate(
            rotate_degree, Image.NEAREST)


class RandomScaleCrop(object):
    def __init__(self, input_size, scale_factor):
        self.input_size = input_size
        self.scale_factor = scale_factor

    def __call__(self, img, mask):
        # random scale (short edge)
        assert img.size[0] == self.input_size

        o_size = np.random.randint(int(self.input_size * 1), int(self.input_size * self.scale_factor))
        img = img.resize((o_size, o_size), resample=Image.BILINEAR)
        mask = mask.resize((o_size, o_size), resample=Image.NEAREST) 

        # random crop input_size
        x1 = np.random.randint(0, o_size - self.input_size)
        y1 = np.random.randint(0, o_size - self.input_size)
        img = img.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))
        mask = mask.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))

        return img, mask


class ScaleCenterCrop(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, mask):
        w, h = img.size
        if w > h:
            oh = self.input_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.input_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), resample=Image.BILINEAR)
        mask = mask.resize((ow, oh), resample=Image.NEAREST)

        w, h = img.size
        x1 = int(round((w - self.input_size) / 2.0))
        y1 = int(round((h - self.input_size) / 2.0))
        img = img.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))
        mask = mask.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))

        return img, mask


class RandomGaussianBlur(object):
    def __call__(self, img, mask):
        if np.random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.random()))

        return img, mask

class AddPepperNoise(object):

    def __init__(self, snr):
        assert isinstance(snr, float)
        self.snr = snr

    def __call__(self, img, gt):
        if np.random.random() < 0.5:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr 
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 
            img_[mask == 2] = 0  
            return Image.fromarray(img_.astype('uint8')).convert('RGB'), gt
        else:
            return img, gt
            
class RandomCrop(object):
    def __call__(self, image, gt):
        image = np.array(image)
        gt = np.array(gt)
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        image = Image.fromarray(image[p0:p1, p2:p3, :])
        gt    = Image.fromarray(gt[p0:p1, p2:p3].astype('uint8'))   

        return image, gt


####################################################################

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        ###--------train---------###
        self.joint_transform_train = Compose([
            RandomHorizontallyFlip(),
            RandomVerticallyFlip(),
            #AddPepperNoise(0.8),
            RandomCrop(),
            # RandomRotate()
        ])  
        self.image_transform_train = transforms.Compose([
            #transforms.ColorJitter(0.1, 0.1, 0.1),  
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.4669, 0.4669, 0.4669], [0.2437, 0.2437, 0.2437])
        ])
        self.mask_transform_train = transforms.ToTensor()  # ->(C,H,W),(0~1)

        ###----------test----------###
        self.image_transform_test = transforms.Compose([
            transforms.Resize((self.cfg.imgsize, self.cfg.imgsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.4669, 0.4669, 0.4669], [0.2437, 0.2437, 0.2437])
        ])
        
        self.mask_transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.samples = os.listdir(cfg.datapath + '/image/')
        # with open(cfg.datapath + '/train.txt', 'r') as lines:
        #     self.samples = []
        #     for line in lines:
        #         self.samples.append(line.strip())

    def __getitem__(self, idx):
        name = self.samples[idx]
        name = name.split('.')[0]
        
        if 'SD900' in self.cfg.datapath.split('/'):
            image_format = '.bmp'
            l_name = name.split('_')[0] + '_' + name.split('_')[-1]
        elif 'ESDIs' in self.cfg.datapath.split('/'):
            image_format = '.jpg'
            l_name = name
        else:
            print('dataname error!')    
        image = Image.open(self.cfg.datapath + '/image/' + name + image_format).convert('RGB')
        
        if self.cfg.mode == 'train':
        
            gt = Image.open(self.cfg.datapath + '/gt/' + l_name + '.png').convert('L')
            
            if image.size == gt.size:
                pass
            else:
                print(image.size, gt.size, name)
            image, gt = self.joint_transform_train(image, gt)
            image = self.image_transform_train(image)
            gt    = self.mask_transform_train(gt)
            return image, gt
        else:
            gt    = Image.open(self.cfg.datapath + '/gt/' + name + '.png').convert('L')
            shape = image.size
            image = self.image_transform_test(image)
            gt    = self.mask_transform_test(gt)
            return image, shape, name, gt

    def __len__(self):
        return len(self.samples)

    def collate(self, batch):  
        # size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]  # 5 scale
        size = self.cfg.imgsize
        image, gt  = [list(item) for item in zip(*batch)]  
        for i in range(len(batch)):  
            image[i] = np.array(image[i]).transpose((1,2,0))
            gt[i]   = np.array(gt[i]).transpose((1,2,0))
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            gt[i] = cv2.resize(gt[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        gt   = torch.from_numpy(np.stack(gt, axis=0)).unsqueeze(dim=1)
        return image, gt


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cfg = Config(mode='train', datapath='./DATA/DUTS/DUTS-TR')
    data = Data(cfg)

    for image, mask, edge in data:
        image = np.array(image).transpose((1, 2, 0))
        mask = np.array(mask).squeeze()
        edge = np.array(edge).squeeze()

        print(image.shape, type(image))

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(image)  
        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.subplot(1, 3, 3)
        plt.imshow(edge)
        plt.show()
        plt.pause(1)

