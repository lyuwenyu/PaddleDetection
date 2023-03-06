import torch

import json
from collections import namedtuple, OrderedDict
import numpy as np

import tensorrt as trt
# https://developer.nvidia.com/nvidia-tensorrt-download

# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py#L305
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/validator.py#L158
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/ops.py#L17
# http://gitlab.baidu.com/paddle-inference/benchmark/blob/main/trt_util.py#L275

path = None
device = torch.device('cuda:0')

Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

logger = trt.Logger(trt.Logger.INFO)
with open(path, 'rb') as f, trt.Runtime(logger) as runtime:
    meta_len = int.from_bytes(
        f.read(4), byteorder='little')  # read metadata length
    meta = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
    model = runtime.deserialize_cuda_engine(f.read())  # read engine
context = model.create_execution_context()

bindings = OrderedDict()
output_names = []
for i in range(model.num_bindings):
    name = model.get_binding_name(i)
    dtype = trt.nptype(model.get_binding_dtype(i))

    if model.binding_is_input(i):
        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
            dynamic = True
            context.set_binding_shape(i,
                                      tuple(model.get_profile_shape(0, i)[2]))
        if dtype == np.float16:
            fp16 = True

    else:  # output
        output_names.append(name)

    shape = tuple(context.get_binding_shape(i))
    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
batch_size = bindings['images'].shape[
    0]  # if dynamic, this is instead max batch size

# --forward--
if dynamic and im.shape != bindings['images'].shape:
    i = model.get_binding_index('images')
    context.set_binding_shape(i, im.shape)  # reshape if dynamic
    bindings['images'] = bindings['images']._replace(shape=im.shape)

    for name in output_names:
        i = model.get_binding_index(name)
        bindings[name].data.resize_(tuple(context.get_binding_shape(i)))

s = bindings['images'].shape
assert im.shape == s, f"input size {im.shape} {'>' if dynamic else 'not equal to'} max model size {s}"
binding_addrs['images'] = int(im.data_ptr())
context.execute_v2(list(binding_addrs.values()))
y = [bindings[x].data for x in sorted(output_names)]
