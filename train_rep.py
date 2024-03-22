#!/usr/bin/python3
#coding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import datetime
import dataset
from torch.utils.data import DataLoader
from models.MobileOneURepHRF import MobileOneURepHRF_s0 as RepLFP
#from models.RepUVit import FastVitURepHRF_s0 as RepLFP
#from models.RepUEVit import EfficientVitURepHRF_s0 as RepLFP
#from models.EUVit import EUVit as RepLFP
from models.MobileOneURepHRF import reparameterize_model
from utils.tools import *
import numpy as np
import random
from timm.scheduler.cosine_lr import CosineLRScheduler 

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
                
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    

def train(Dataset, Network):

    cfg    = Dataset.Config(datapath='./dataset/ESDIs/train', imgsize=224, pretrain=True, savepath='./model_save/RepLFPNet', mode='train', batch=96, lr=5e-4, momen=0.9, decay=5e-4, epoch=800)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=0)
    print(len(loader))
    ##network
    net = Network(cfg)
    net.train(True)
    net.cuda()
    base, head = [], []
    for name, param in net.named_parameters():
        if 'encoder' in name or 'decoder' in name:
            base.append(param)
        else:
            head.append(param)

    #optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    optimizer      = torch.optim.Adam([{'params':base}, {'params':head}], lr=cfg.lr)
    #scheduler      = CosineLRScheduler(optimizer=optimizer, t_initial=200, lr_min=1e-6, cycle_limit=10, cycle_decay=0.5, warmup_t=10, warmup_lr_init=1e-5)
    scheduler      = CosineLRScheduler(optimizer=optimizer, t_initial=800, lr_min=1e-6, warmup_t=10, warmup_lr_init=1e-5)
    global_step    = 0


    CE = torch.nn.BCELoss().cuda()
    ssim_loss = SSIM(window_size=11,size_average=True)
    iou_loss = IOU(size_average=True)
    def hybrid_loss(pred,target):
        bce_out = CE(pred,target)
        iou_out = iou_loss(pred,target)
        ssim_out = 1 - ssim_loss(pred,target)
        loss = bce_out + ssim_out + iou_out
        return loss

    for epoch in range(cfg.epoch):
        #optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.5
        #optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr
        scheduler.step(epoch)
        for step, (image, gt) in enumerate(loader):
            image, gt = image.type(torch.FloatTensor).cuda(), gt.type(torch.FloatTensor).cuda()                              
                                                                                                                    
            sout1, sout2, sout3, out = net(image)
            
            loss = hybrid_loss(torch.sigmoid(out), gt)
              
            optimizer.zero_grad()
            loss.backward() 
            clip_gradient(optimizer, cfg.lr)
            optimizer.step()

            global_step += 1
            if step%10 == 0:
                print('%s | step:%d/%d/%d | base_lr=%.6f | loss_hyb=%.6f'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], 
                       loss.item()))


        if not os.path.exists(cfg.savepath):
            os.makedirs(cfg.savepath)
            
        if (epoch+1)%50==0 and epoch > (cfg.epoch//2):
            repnet = reparameterize_model(net)
            torch.save(repnet.state_dict(), cfg.savepath + '/model-'+str(epoch+1)+'.pth')

                
        if (epoch+1) == cfg.epoch:
            repnet = reparameterize_model(net)
            torch.save(repnet.state_dict(), cfg.savepath+ '/model-'+str(epoch+1)+'.pth')
            

if __name__=='__main__':
    set_seed(7)
    train(dataset, RepLFP)


