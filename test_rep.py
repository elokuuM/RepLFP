#!/usr/bin/python3
#coding=utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import tensorrt as trt
import onnx, onnxsim, onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
import dataset
import time
import cv2
from torch.utils.data import DataLoader
from models.MobileOneURepHRF import MobileOneURepHRF_s0 as RepLFP
import timeit
import netron
from typing import Union, Optional, Sequence, Dict, Any


class TRTWrapper(torch.nn.Module):
    def __init__(self, engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

            # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs


def export_onnx_graph(net):
    input  = torch.Tensor(1, 3, 224, 224)
    model  = net
    model.eval()

    file   = "./model_save/onnx/model.onnx"
    torch.onnx.export(
            model         = model,
            args          = input,
            f             = file,
            input_names   = ["input0"],
            output_names  = ["output0"],
            opset_version = 11)

    print("\nFinished export {}".format(file))

    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    #netron.start(file)
    assert check, "assert onnxsim check failed"
    onnx.save(model_onnx, file)

class Test(object):
    def __init__(self, Dataset, Network, Path, snapshot):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, imgsize=224, snapshot=snapshot, mode='test') 
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        

    
        self.net = Network(self.cfg)  
        trained_dict = torch.load(self.cfg.snapshot,map_location=torch.device('cpu'))
        print('load repnet', self.net.load_state_dict(trained_dict))

        export_onnx_graph(self.net)
        self.net.train(False)
        self.net.cuda()
        

    def save(self):
        with torch.no_grad():
            cost_time = 0
            avg_mae = 0.0
            cnt = 0
            for image, (H, W), name, gt in self.loader:
                start_time = time.perf_counter()
                image, shape, gt = image.cuda().float(), (H, W), gt.cuda().float()
                
                out = self.net(image)
                cost_time += time.perf_counter() - start_time
                
                out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
                pred = torch.sigmoid(out[0, 0])

                avg_mae += torch.abs(pred - gt).mean().item()
                cnt += len(image)
                pred = pred.cpu().numpy() * 255

                head = './test_out/replfp'
                
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.uint8(pred))

            fps = len(self.loader.dataset) / cost_time
            msg = '%s len(imgs)=%s, fps=%.4f'%(self.cfg.datapath, len(self.loader.dataset), fps)              
            print(msg)
            print('num_image:', cnt)
            print('MAE:', avg_mae / cnt)

    def onnx_infer_time(self):
        with torch.no_grad():
            input = torch.rand(1, 3, 224, 224)
            torch_model = self.net.cpu()
            onnx_model = onnxruntime.InferenceSession('./model_save/onnx/replfp.onnx',
                                                      providers=['CPUExecutionProvider'])

            # pth模型推理
            def torch_inf():
                x = torch_model(input)
                return x
            # onnx模型跑推理
            def onnx_inf():
                x = onnx_model.run(None, {
                    onnx_model.get_inputs()[0].name: input.numpy()})
                return x

            # 设置循环次数
            n = 100

            torch_time = timeit.timeit(lambda: torch_inf(), number=n) / n  # cpu: 0.0246

            onnx_time = timeit.timeit(lambda: onnx_inf(), number=n) / n  # cpu: 0.00291

            print(torch_time, onnx_time)
            print(torch_inf(), onnx_inf())

    def onnx_test(self):
        with torch.no_grad():
            cost_time = 0
            avg_mae = 0.0
            cnt = 0
            sess_options = onnxruntime.SessionOptions()
            model = onnx.load('./model_save/onnx/replfp.onnx')
            # 'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'
            onnx_model = onnxruntime.InferenceSession(model.SerializeToString(), sess_options,
                                                      providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

            for image, (H, W), name, gt in self.loader:

                image, shape, gt = image.float(), (H, W), gt.float()
                start_time = time.perf_counter()
                out = onnx_model.run(None, {onnx_model.get_inputs()[0].name: image.numpy()})
                cost_time += time.perf_counter() - start_time

                out = torch.tensor(out[0])
                out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
                pred = torch.sigmoid(out[0, 0])

                avg_mae += torch.abs(pred - gt).mean().item()
                cnt += len(image)
                pred = pred.cpu().numpy() * 255

            fps = len(self.loader.dataset) / cost_time
            msg = '%s len(imgs)=%s, fps=%.4f' % (self.cfg.datapath, len(self.loader.dataset), fps)
            print(msg)
            print('num_image:', cnt)
            print('MAE:', avg_mae / cnt)

    def onnxTtrt(self):

        file = "./model_save/onnx/replfp.onnx"
        onnx_model = onnx.load(file)

        # create builder and network
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        EXPLICIT_BATCH = 1 << (int)(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(EXPLICIT_BATCH)

        # parse onnx
        parser = trt.OnnxParser(network, logger)

        if not parser.parse(onnx_model.SerializeToString()):
            error_msgs = ''
            for error in range(parser.num_errors):
                error_msgs += f'{parser.get_error(error)}\n'
            raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 20
        profile = builder.create_optimization_profile()

        profile.set_shape('input', [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])
        config.add_optimization_profile(profile)
        # create engine
        device = torch.device('cuda:0')
        with torch.cuda.device(device):
            engine = builder.build_engine(network, config)

        save_file = './model_save/trt/'
        if not os.path.exists(save_file):
            os.makedirs(save_file)
        with open(save_file + 'model.engine', mode='wb') as f:
            f.write(bytearray(engine.serialize()))
            print("generating file done!")

    def trt_test(self):
        with torch.no_grad():
            cost_time = 0
            avg_mae = 0.0
            cnt = 0
            model = TRTWrapper('./model_save/trt/model.engine', ['output0']) #输入输出需与onnx定义一致

            for image, (H, W), name, gt in self.loader:

                image, shape, gt = image.float().cuda(), (H, W), gt.float().cuda()
                start_time = time.perf_counter()
                out = model(dict({'input0': image})) #out是一个字典
                cost_time += time.perf_counter() - start_time

                out = out['output0']
                out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
                pred = torch.sigmoid(out[0, 0])

                avg_mae += torch.abs(pred - gt).mean().item()
                cnt += len(image)
                pred = pred.cpu().numpy() * 255

            fps = len(self.loader.dataset) / cost_time
            msg = '%s len(imgs)=%s, fps=%.4f' % (self.cfg.datapath, len(self.loader.dataset), fps)
            print(msg)
            print('num_image:', cnt)
            print('MAE:', avg_mae / cnt)

if __name__=='__main__':
    for path in ['./dataset/ESDIs/test']:
        t = Test(dataset, RepLFP, path, './model_save/model-replfp.pth')
        t.save()
        #t.onnx_infer_time()
        #t.onnx_test()   #1.26-fold_cuda  3.3-fold_trt  8.4-fold_cpu
        #t.onnxTtrt()
        #t.trt_test()    #6.2-fold