
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from torchstat import stat
from thop import profile

from timm.scheduler.cosine_lr import CosineLRScheduler 
from models.backbone.MobileOne_Repmlp import RepUnet, reparameterize_model
from models.MobileOneURepHRF import MobileOneURepHRF_s0
torch.backends.cudnn.benchmark = True



class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None
cfg    = Config(mode='test')

################Net-set############################
Net = MobileOneURepHRF_s0(cfg)
Net = reparameterize_model(Net)

Net = Net.cuda()
input_size = 224
################test-IO############################
inputs = torch.rand(1, 3, input_size, input_size).cuda()
repout = Net(inputs)
#print('Net out shape', repout.shape)

################test-fps###########################
def computeTime(model, device='cuda', batchs=100):
    inputs = torch.randn(batchs, 3, input_size, input_size)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()
    else:
        model = model.cpu()
        inputs = inputs.cpu()
    model.eval()

    time_spent = []
    for idx in tqdm(range(100)):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Average speed on {}: {:.4f} fps'.format(device, batchs / np.mean(time_spent)))

computeTime(Net, device='cuda',batchs=200)
computeTime(Net, device='cpu',batchs=1)

################test-param#########################
#stat(Net.cpu(), (3, 224, 224))
input = torch.randn(1, 3, input_size, input_size).cuda()
flops, params = profile(Net.cuda(), inputs=(input, ))
print('params:%.2f(M)'%(params/1e6))
print('flops:%.2f(G)'%(flops/1e9))


