import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import copy
from models.backbone.MobileOne_Repmlp import MobileOneBlock
from models.backbone.MobileNetV2 import mobilenetv2


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1):
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
    result.add_module('relu', nn.ReLU())
    return result
    
    

        
class RepHRFBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation = [1, 2, 3, 4],
                 groups: int = 1,
                 expand_ratio = 2.,
                 num_conv_branches: int = 4,
                 inference_mode: bool = False) -> None:
        """ Construct a RepAsppBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(RepHRFBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.dilation = dilation
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = int(in_channels * expand_ratio)
        self.num_conv_branches = num_conv_branches
        
        self.activation = nn.ReLU()
        self.squeezeconv = MobileOneBlock(in_channels=self.in_channels,
                                         out_channels=self.hid_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         num_conv_branches=self.num_conv_branches) 
                                         
        self.invertconv = MobileOneBlock(in_channels=self.hid_channels,
                                         out_channels=self.in_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                        num_conv_branches=self.num_conv_branches)  
                                         
        self.channelconv = MobileOneBlock(in_channels=self.hid_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         num_conv_branches=self.num_conv_branches)                             

        if inference_mode:
            self.reparam_conv = nn.ModuleList()
            for idx in range(len(self.dilation)):
                self.reparam_conv.append(nn.Conv2d(in_channels=self.hid_channels,
                                              out_channels=self.hid_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=self.dilation[idx] * (kernel_size//2),
                                              dilation=self.dilation[idx],
                                              groups=self.hid_channels,
                                              bias=True))
                                                                                        
        else:
        
            self.rbr_skip = nn.ModuleList()
            self.rbr_conv = nn.ModuleList()
            self.rbr_scale = nn.ModuleList()
            # Re-parameterizable skip connection
            for idx in range(len(self.dilation)):
                self.rbr_skip.append(nn.BatchNorm2d(num_features=self.hid_channels))

            # Re-parameterizable conv branches
            for idx in range(len(self.dilation)):
                for _ in range(self.num_conv_branches):
                    self.rbr_conv.append(conv_bn(in_channels=self.hid_channels,
                                                 out_channels=self.hid_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride, 
                                                 padding=self.dilation[idx] * (kernel_size//2),
                                                 dilation=self.dilation[idx], 
                                                 groups=self.hid_channels))
            # Re-parameterizable scale branch
            for idx in range(len(self.dilation)):
                if kernel_size > 1:
                    self.rbr_scale.append(conv_bn(in_channels=self.hid_channels,
                                             out_channels=self.hid_channels,
                                             kernel_size=1,
                                             stride=1, 
                                             padding=0,
                                             dilation=1, 
                                             groups=self.hid_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        out = 0
        short_cut = x
        if self.inference_mode:
            x = self.squeezeconv(x)
            for idx in range(len(self.dilation)):
                out = out + self.activation(self.reparam_conv[idx](x))
                
           #out = self.invertconv(out) + short_cut
            return self.channelconv(out)

        # Multi-branched train-time forward pass.

        x = self.squeezeconv(x)
        for idx in range(len(self.dilation)):
            # Skip branch output
            identity_out = 0
            if self.rbr_skip is not None:
                identity_out = self.rbr_skip[idx](x)
    
            # Scale branch output
            scale_out = 0
            if self.rbr_scale is not None:
                scale_out = self.rbr_scale[idx](x)
    
            # Other branches
            conv_out = 0
            for ix in range(self.num_conv_branches):
                conv_out += self.rbr_conv[(idx*self.num_conv_branches)+ix](x)

            out = out + self.activation(identity_out + scale_out + conv_out)            
        
 
        #out = self.invertconv(out) + short_cut
        return self.channelconv(out)

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        self.reparam_conv = nn.ModuleList()
        for idx in range(len(self.dilation)):
            kernel, bias = self._get_kernel_bias(idx)
            self.reparam_conv.append(nn.Conv2d(in_channels=self.rbr_conv[idx * self.num_conv_branches].conv.in_channels,
                                          out_channels=self.rbr_conv[idx * self.num_conv_branches].conv.out_channels,
                                          kernel_size=self.rbr_conv[idx * self.num_conv_branches].conv.kernel_size,
                                          stride=self.rbr_conv[idx * self.num_conv_branches].conv.stride,
                                          padding=self.rbr_conv[idx * self.num_conv_branches].conv.padding,
                                          dilation=self.rbr_conv[idx * self.num_conv_branches].conv.dilation,
                                          groups=self.rbr_conv[idx * self.num_conv_branches].conv.groups,
                                          bias=True))
            self.reparam_conv[idx].weight.data = kernel
            self.reparam_conv[idx].bias.data = bias
        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True
        print('repaspp reparameterize done!')
    def _get_kernel_bias(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale[idx] is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale[idx])
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip[idx] is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip[idx])

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[(idx*self.num_conv_branches)+ix])
            kernel_conv += _kernel
            bias_conv += _bias
        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.hid_channels // self.hid_channels
                kernel_value = torch.zeros((self.hid_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.hid_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list

def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model
    
class RepHRFdecoder(nn.Module):         #enc_channels=[48, 48, 128, 256]  #(16, 24, 32, 96, 160)
    def __init__(self, inference_mode = False, enc_channels = [16, 24, 32, 96, 160], expand_ratio = 0.5):
        super().__init__()
        self.inference_mode = inference_mode
        
        self.deoder0 = RepHRFBlock(enc_channels[-1], enc_channels[-2],
                                   dilation = [1, 2, 3],  #[1, 2, 3, 4]     [1, 2]
                                   expand_ratio = expand_ratio,
                                   inference_mode = self.inference_mode)
                                    
        self.deoder1 = RepHRFBlock(enc_channels[-2], enc_channels[-3],
                                   dilation = [1, 2, 3, 4],  #[1, 2, 3, 4]     [1, 2]
                                   expand_ratio = expand_ratio,
                                   inference_mode = self.inference_mode)
                       
        self.deoder2 = RepHRFBlock(enc_channels[-3], enc_channels[-4],
                                   dilation = [1, 2, 4, 8],  #[1, 2, 4, 8]     [1, 2, 3, 4]
                                   expand_ratio = expand_ratio,
                                   inference_mode = self.inference_mode)
                       
        self.deoder3 = RepHRFBlock(enc_channels[-4], enc_channels[-5],
                                   dilation = [1, 4, 8, 16],  #[1, 4, 8, 16]    [1, 2, 4, 8]
                                   expand_ratio = expand_ratio,
                                   inference_mode = self.inference_mode)
                       
        self.deoder4 = RepHRFBlock(enc_channels[-5], 32,
                                   dilation = [1, 4, 8, 16],  #[1, 4, 8, 16]    [1, 4, 8, 16]
                                   expand_ratio = expand_ratio,
                                   inference_mode = self.inference_mode)
        
        self.convout = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1),nn.BatchNorm2d(16), nn.ReLU(inplace=True),nn.Conv2d(16, 1, 1, 1, padding=0))
                                    
        self.side_out1  = nn.Conv2d(enc_channels[-3], 1, kernel_size=1, stride=1, padding=0)
        self.side_out2  = nn.Conv2d(enc_channels[-4], 1, kernel_size=1, stride=1, padding=0)
        self.side_out3  = nn.Conv2d(enc_channels[-5], 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        [out1, out2, out3, out4, out5] = x
        if self.inference_mode:
            out = F.interpolate(self.deoder0(out5), size=out4.shape[-1], mode='bilinear', align_corners=False)  + out4 # 14
            out = F.interpolate(self.deoder1(out), size=out3.shape[-1], mode='bilinear', align_corners=False)   + out3 # 28
            out = F.interpolate(self.deoder2(out), size=out2.shape[-1], mode='bilinear', align_corners=False)   + out2 # 56
            out = F.interpolate(self.deoder3(out), size=out1.shape[-1], mode='bilinear', align_corners=False)   + out1 # 112
            out = self.convout(self.deoder4(out))
            out = F.interpolate(out, size=out1.shape[-1] * 2, mode='bilinear', align_corners=False)
            return out
        else:
            out = F.interpolate(self.deoder0(out5), size=out4.shape[-1], mode='bilinear', align_corners=False)  + out4 # 14
        
            out = F.interpolate(self.deoder1(out), size=out3.shape[-1], mode='bilinear', align_corners=False)   + out3 # 28
            sout1 = F.interpolate(self.side_out1(out), size=out1.shape[-1] * 2, mode='bilinear', align_corners=False)
            
            out = F.interpolate(self.deoder2(out), size=out2.shape[-1], mode='bilinear', align_corners=False)   + out2 # 56
            sout2 = F.interpolate(self.side_out2(out), size=out1.shape[-1] * 2, mode='bilinear', align_corners=False)
            
            out = F.interpolate(self.deoder3(out), size=out1.shape[-1], mode='bilinear', align_corners=False)   + out1 # 112
            sout3 = F.interpolate(self.side_out3(out), size=out1.shape[-1] * 2, mode='bilinear', align_corners=False)
            
            
            out = self.convout(self.deoder4(out))
            out = F.interpolate(out, size=out1.shape[-1] * 2, mode='bilinear', align_corners=False)
            return sout1, sout2, sout3, out
        
class MobileNetv2URepHRF(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        if cfg.mode == 'train':
            self.encoder = mobilenetv2(pretrained=True)
            self.decoder = RepHRFdecoder(inference_mode = False, enc_channels = [16, 24, 32, 96, 160], expand_ratio = 2)
        elif cfg.mode == 'test':
            self.encoder = mobilenetv2(pretrained=True)
            self.decoder = RepHRFdecoder(inference_mode = True, enc_channels = [16, 24, 32, 96, 160], expand_ratio = 2)
        else:
            print('mode error!')
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


