# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import weakref
import paddle
import paddle.nn as nn

import paddle.optimizer as optimizer
import paddle.regularizer as regularizer

from ppdet.core.workspace import register, serializable

__all__ = ['LearningRate', 'OptimizerBuilder']

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@serializable
class CosineDecay(object):
    """
    Cosine learning rate decay

    Args:
        max_epochs (int): max epochs for the training process.
            if you commbine cosine decay with warmup, it is recommended that
            the max_iters is much larger than the warmup iter
        use_warmup (bool): whether to use warmup. Default: True.
        min_lr_ratio (float): minimum learning rate ratio. Default: 0.
        last_plateau_epochs (int): use minimum learning rate in
            the last few epochs. Default: 0.
    """

    def __init__(self,
                 max_epochs=1000,
                 use_warmup=True,
                 min_lr_ratio=0.,
                 last_plateau_epochs=0):
        self.max_epochs = max_epochs
        self.use_warmup = use_warmup
        self.min_lr_ratio = min_lr_ratio
        self.last_plateau_epochs = last_plateau_epochs

    def __call__(self,
                 base_lr=None,
                 boundary=None,
                 value=None,
                 step_per_epoch=None):
        assert base_lr is not None, "either base LR or values should be provided"

        max_iters = self.max_epochs * int(step_per_epoch)
        last_plateau_iters = self.last_plateau_epochs * int(step_per_epoch)
        min_lr = base_lr * self.min_lr_ratio
        if boundary is not None and value is not None and self.use_warmup:
            # use warmup
            warmup_iters = len(boundary)
            for i in range(int(boundary[-1]), max_iters):
                boundary.append(i)
                if i < max_iters - last_plateau_iters:
                    decayed_lr = min_lr + (base_lr - min_lr) * 0.5 * (math.cos(
                        (i - warmup_iters) * math.pi /
                        (max_iters - warmup_iters - last_plateau_iters)) + 1)
                    value.append(decayed_lr)
                else:
                    value.append(min_lr)
            return optimizer.lr.PiecewiseDecay(boundary, value)
        elif last_plateau_iters > 0:
            # not use warmup, but set `last_plateau_epochs` > 0
            boundary = []
            value = []
            for i in range(max_iters):
                if i < max_iters - last_plateau_iters:
                    decayed_lr = min_lr + (base_lr - min_lr) * 0.5 * (math.cos(
                        i * math.pi / (max_iters - last_plateau_iters)) + 1)
                    value.append(decayed_lr)
                else:
                    value.append(min_lr)
                if i > 0:
                    boundary.append(i)
            return optimizer.lr.PiecewiseDecay(boundary, value)

        return optimizer.lr.CosineAnnealingDecay(
            base_lr, T_max=max_iters, eta_min=min_lr)


@serializable
class PiecewiseDecay(object):
    """
    Multi step learning rate decay

    Args:
        gamma (float | list): decay factor
        milestones (list): steps at which to decay learning rate
    """

    def __init__(self,
                 gamma=[0.1, 0.01],
                 milestones=[8, 11],
                 values=None,
                 use_warmup=True):
        super(PiecewiseDecay, self).__init__()
        if type(gamma) is not list:
            self.gamma = []
            for i in range(len(milestones)):
                self.gamma.append(gamma / 10**i)
        else:
            self.gamma = gamma
        self.milestones = milestones
        self.values = values
        self.use_warmup = use_warmup

    def __call__(self,
                 base_lr=None,
                 boundary=None,
                 value=None,
                 step_per_epoch=None):
        if boundary is not None and self.use_warmup:
            boundary.extend([int(step_per_epoch) * i for i in self.milestones])
        else:
            # do not use LinearWarmup
            boundary = [int(step_per_epoch) * i for i in self.milestones]
            value = [base_lr]  # during step[0, boundary[0]] is base_lr

        # self.values is setted directly in config
        if self.values is not None:
            assert len(self.milestones) + 1 == len(self.values)
            return optimizer.lr.PiecewiseDecay(boundary, self.values)

        # value is computed by self.gamma
        value = value if value is not None else [base_lr]
        for i in self.gamma:
            value.append(base_lr * i)

        return optimizer.lr.PiecewiseDecay(boundary, value)


@serializable
class LinearWarmup(object):
    """
    Warm up learning rate linearly

    Args:
        steps (int): warm up steps
        start_factor (float): initial learning rate factor
        epochs (int|None): use epochs as warm up steps, the priority
            of `epochs` is higher than `steps`. Default: None.
    """

    def __init__(self, steps=500, start_factor=1. / 3, epochs=None):
        super(LinearWarmup, self).__init__()
        self.steps = steps
        self.start_factor = start_factor
        self.epochs = epochs

    def __call__(self, base_lr, step_per_epoch):
        boundary = []
        value = []
        warmup_steps = self.epochs * step_per_epoch \
            if self.epochs is not None else self.steps
        for i in range(warmup_steps + 1):
            if warmup_steps > 0:
                alpha = i / warmup_steps
                factor = self.start_factor * (1 - alpha) + alpha
                lr = base_lr * factor
                value.append(lr)
            if i > 0:
                boundary.append(i)
        return boundary, value


@serializable
class BurninWarmup(object):
    """
    Warm up learning rate in burnin mode
    Args:
        steps (int): warm up steps
    """

    def __init__(self, steps=1000):
        super(BurninWarmup, self).__init__()
        self.steps = steps

    def __call__(self, base_lr, step_per_epoch):
        boundary = []
        value = []
        burnin = min(self.steps, step_per_epoch)
        for i in range(burnin + 1):
            factor = (i * 1.0 / burnin)**4
            lr = base_lr * factor
            value.append(lr)
            if i > 0:
                boundary.append(i)
        return boundary, value


@serializable
class ExpWarmup(object):
    """
    Warm up learning rate in exponential mode
    Args:
        steps (int): warm up steps.
        epochs (int|None): use epochs as warm up steps, the priority
            of `epochs` is higher than `steps`. Default: None.
    """

    def __init__(self, steps=5, epochs=None):
        super(ExpWarmup, self).__init__()
        self.steps = steps
        self.epochs = epochs

    def __call__(self, base_lr, step_per_epoch):
        boundary = []
        value = []
        warmup_steps = self.epochs * step_per_epoch if self.epochs is not None else self.steps
        for i in range(warmup_steps + 1):
            factor = (i / float(warmup_steps))**2
            value.append(base_lr * factor)
            if i > 0:
                boundary.append(i)
        return boundary, value


@register
class LearningRate(object):
    """
    Learning Rate configuration

    Args:
        base_lr (float): base learning rate
        schedulers (list): learning rate schedulers
    """
    __category__ = 'optim'

    def __init__(self,
                 base_lr=0.01,
                 schedulers=[PiecewiseDecay(), LinearWarmup()]):
        super(LearningRate, self).__init__()
        self.base_lr = base_lr
        self.schedulers = schedulers

    def __call__(self, step_per_epoch):
        assert len(self.schedulers) >= 1
        if not self.schedulers[0].use_warmup:
            return self.schedulers[0](base_lr=self.base_lr,
                                      step_per_epoch=step_per_epoch)

        # TODO: split warmup & decay
        # warmup
        boundary, value = self.schedulers[1](self.base_lr, step_per_epoch)
        # decay
        decay_lr = self.schedulers[0](self.base_lr, boundary, value,
                                      step_per_epoch)
        return decay_lr


@register
class OptimizerBuilder():
    """
    Build optimizer handles
    Args:
        regularizer (object): an `Regularizer` instance
        optimizer (object): an `Optimizer` instance
    """
    __category__ = 'optim'

    def __init__(self,
                 clip_grad_by_norm=None,
                 regularizer={'type': 'L2',
                              'factor': .0001},
                 optimizer={'type': 'Momentum',
                            'momentum': .9}):
        self.clip_grad_by_norm = clip_grad_by_norm
        self.regularizer = regularizer
        self.optimizer = optimizer

    def __call__(self, learning_rate, model=None):
        if self.clip_grad_by_norm is not None:
            grad_clip = nn.ClipGradByGlobalNorm(
                clip_norm=self.clip_grad_by_norm)
        else:
            grad_clip = None
        if self.regularizer and self.regularizer != 'None':
            reg_type = self.regularizer['type'] + 'Decay'
            reg_factor = self.regularizer['factor']
            regularization = getattr(regularizer, reg_type)(reg_factor)
        else:
            regularization = None

        optim_args = self.optimizer.copy()
        optim_type = optim_args['type']
        del optim_args['type']
        if optim_type != 'AdamW':
            optim_args['weight_decay'] = regularization
        op = getattr(optimizer, optim_type)

        if 'param_groups' in optim_args:
            assert isinstance(optim_args['param_groups'], list), ''

            param_groups = optim_args.pop('param_groups')

            params, visited = [], []
            for group in param_groups:
                assert isinstance(group,
                                  dict) and 'params' in group and isinstance(
                                      group['params'], list), ''
                _params = {
                    n: p
                    for n, p in model.named_parameters()
                    if any([k in n for k in group['params']])
                }
                _group = group.copy()
                _group.update({'params': list(_params.values())})

                params.append(_group)
                visited.extend(list(_params.keys()))

            ext_params = [
                p for n, p in model.named_parameters() if n not in visited
            ]

            if len(ext_params) < len(model.parameters()):
                params.append({'params': ext_params})

            elif len(ext_params) > len(model.parameters()):
                raise RuntimeError

        else:
            params = model.parameters()

        return op(learning_rate=learning_rate,
                  parameters=params,
                  grad_clip=grad_clip,
                  **optim_args)


class ModelEMA(object):
    """
    Exponential Weighted Average for Deep Neutal Networks
    Args:
        model (nn.Layer): Detector of model.
        decay (int):  The decay used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `ema_param = decay * ema_param + (1 - decay) * cur_param`.
            Defaults is 0.9998.
        ema_decay_type (str): type in ['threshold', 'normal', 'exponential'],
            'threshold' as default.
        cycle_epoch (int): The epoch of interval to reset ema_param and
            step. Defaults is -1, which means not reset. Its function is to
            add a regular effect to ema, which is set according to experience
            and is effective when the total training epoch is large.
    """

    def __init__(self,
                 model,
                 decay=0.9998,
                 ema_decay_type='threshold',
                 cycle_epoch=-1):
        self.step = 0
        self.epoch = 0
        self.decay = decay
        self.state_dict = dict()
        for k, v in model.state_dict().items():
            self.state_dict[k] = paddle.zeros_like(v)
        self.ema_decay_type = ema_decay_type
        self.cycle_epoch = cycle_epoch

        self._model_state = {
            k: weakref.ref(p)
            for k, p in model.state_dict().items()
        }

    def reset(self):
        self.step = 0
        self.epoch = 0
        for k, v in self.state_dict.items():
            self.state_dict[k] = paddle.zeros_like(v)

    def resume(self, state_dict, step=0):
        for k, v in state_dict.items():
            if k in self.state_dict:
                self.state_dict[k] = v
        self.step = step

    def update(self, model=None):
        if self.ema_decay_type == 'threshold':
            decay = min(self.decay, (1 + self.step) / (10 + self.step))
        elif self.ema_decay_type == 'exponential':
            decay = self.decay * (1 - math.exp(-(self.step + 1) / 2000))
        else:
            decay = self.decay
        self._decay = decay

        if model is not None:
            model_dict = model.state_dict()
        else:
            model_dict = {k: p() for k, p in self._model_state.items()}
            assert all(
                [v is not None for _, v in model_dict.items()]), 'python gc.'

        for k, v in self.state_dict.items():
            v = decay * v + (1 - decay) * model_dict[k]
            v.stop_gradient = True
            self.state_dict[k] = v
        self.step += 1

    def apply(self):
        if self.step == 0:
            return self.state_dict
        state_dict = dict()
        for k, v in self.state_dict.items():
            if self.ema_decay_type != 'exponential':
                v = v / (1 - self._decay**self.step)
            v.stop_gradient = True
            state_dict[k] = v
        self.epoch += 1
        if self.cycle_epoch > 0 and self.epoch == self.cycle_epoch:
            self.reset()

        return state_dict


def layerwise_lr_decay(decay_rate, name_dict, n_layers, param):
    """
    Args:
        decay_rate (float): 
            The layer-wise decay ratio.
        name_dict (dict): 
            The keys of name_dict is dynamic name of model while the value
            of name_dict is static name.
            Use model.named_parameters() to get name_dict.
        n_layers (int):
            Total number of layers in the transformer encoder.
    """
    ratio = 1.0
    static_name = name_dict[param.name]
    if "blocks" in static_name:
        idx = static_name.find("blocks.")
        layer = int(static_name[idx:].split(".")[1])
        ratio = decay_rate**(n_layers - layer)

    # TODO same as pytorch, cls_token and patch_embed lr = 0.75 ** (12 + 1)

    # elif "embed" in static_name:
    # elif "patch_embed" in static_name:

    elif "cls_token" in static_name or 'patch_embed' in static_name:
        ratio = decay_rate**(n_layers + 1)

    param.optimize_attr["learning_rate"] *= ratio


from paddle.optimizer import AdamW
import paddle.fluid as fluid
from functools import partial


class AdamWDL(AdamW):
    r"""
    The AdamWDL optimizer is implemented based on the AdamW Optimization with dynamic lr setting.
    Generally it's used for transformer model.

    We use "layerwise_lr_decay" as default dynamic lr setting method of AdamWDL.
    “Layer-wise decay” means exponentially decaying the learning rates of individual 
    layers in a top-down manner. For example, suppose the 24-th layer uses a learning
    rate l, and the Layer-wise decay rate is α, then the learning rate of layer m 
    is lα^(24-m). See more details on: https://arxiv.org/abs/1906.08237.

    .. math::
        & t = t + 1
    
        & moment\_1\_out = {\beta}_1 * moment\_1 + (1 - {\beta}_1) * grad

        & moment\_2\_out = {\beta}_2 * moment\_2 + (1 - {\beta}_2) * grad * grad

        & learning\_rate = learning\_rate * \frac{\sqrt{1 - {\beta}_2^t}}{1 - {\beta}_1^t}

        & param\_out = param - learning\_rate * (\frac{moment\_1}{\sqrt{moment\_2} + \epsilon} + \lambda * param)

    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 1e-08.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float, optional): The weight decay coefficient, it can be float or Tensor. The default value is 0.01.
        apply_decay_param_fun (function|None, optional): If it is not None,
            only tensors that makes apply_decay_param_fun(Tensor.name)==True
            will be updated. It only works when we want to specify tensors.
            Default: None.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        lazy_mode (bool, optional): The official Adam algorithm has two moving-average accumulators.
            The accumulators are updated at every step. Every element of the two moving-average
            is updated in both dense mode and sparse mode. If the size of parameter is very large,
            then the update may be very slow. The lazy mode only update the element that has
            gradient in current mini-batch, so it will be much more faster. But this mode has
            different semantics with the original Adam algorithm and may lead to different result.
            The default value is False.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating. Default is false.  
        layerwise_decay (float, optional): The layer-wise decay ratio. Defaults to 1.0.
        n_layers (int, optional): The total number of encoder layers. Defaults to 12.
        set_param_lr_fun (function|None, optional): If it's not None, set_param_lr_fun() will set the the parameter 
            learning rate before it executes Adam Operator. Defaults to :ref:`layerwise_lr_decay`.
        name_dict (dict, optional): The keys of name_dict is dynamic name of model while the value
            of name_dict is static name. Use model.named_parameters() to get name_dict.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Examples:
        .. code-block:: python

            import paddle
            from paddlenlp.ops.optimizer import AdamWDL
            def simple_lr_setting(decay_rate, name_dict, n_layers, param):
                ratio = 1.0
                static_name = name_dict[param.name]
                if "weight" in static_name:
                    ratio = decay_rate**0.5
                param.optimize_attr["learning_rate"] *= ratio
            
            linear = paddle.nn.Linear(10, 10)

            name_dict = dict()
            for n, p in linear.named_parameters():
                name_dict[p.name] = n

            inp = paddle.rand([10,10], dtype="float32")
            out = linear(inp)
            loss = paddle.mean(out)

            adamwdl = AdamWDL(
                learning_rate=1e-4,
                parameters=linear.parameters(),
                set_param_lr_fun=simple_lr_setting,
                layerwise_decay=0.8,
                name_dict=name_dict)
            
            loss.backward()
            adamwdl.step()
            adamwdl.clear_grad()
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameters=None,
                 weight_decay=0.01,
                 apply_decay_param_fun=None,
                 grad_clip=None,
                 lazy_mode=False,
                 multi_precision=False,
                 layerwise_decay=1.0,
                 n_layers=12,
                 set_param_lr_fun=layerwise_lr_decay,
                 name_dict=None,
                 name=None):
        if not isinstance(layerwise_decay, float) and \
                not isinstance(layerwise_decay, fluid.framework.Variable):
            raise TypeError("coeff should be float or Tensor.")
        self.layerwise_decay = layerwise_decay
        self.n_layers = n_layers
        self.set_param_lr_fun = partial(set_param_lr_fun, layerwise_decay,
                                        name_dict, n_layers)
        super(AdamWDL, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            grad_clip=grad_clip,
            name=name,
            apply_decay_param_fun=apply_decay_param_fun,
            weight_decay=weight_decay,
            lazy_mode=lazy_mode,
            multi_precision=multi_precision)

    def _append_optimize_op(self, block, param_and_grad):
        if self.set_param_lr_fun is None:
            return super(AdamWDL, self)._append_optimize_op(block,
                                                            param_and_grad)

        self._append_decoupled_weight_decay(block, param_and_grad)
        prev_lr = param_and_grad[0].optimize_attr["learning_rate"]
        self.set_param_lr_fun(param_and_grad[0])
        # excute Adam op
        res = super(AdamW, self)._append_optimize_op(block, param_and_grad)
        param_and_grad[0].optimize_attr["learning_rate"] = prev_lr
        return res


import numpy as np


def polynomial_scheduler(base_value,
                         final_value,
                         total_iters,
                         start_warmup_value=0,
                         warmup_iters=-1):
    '''polynomial_scheduler
    '''
    warmup_schedule = np.array([])
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = total_iters - warmup_iters

    scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=base_value,
        decay_steps=iters,
        end_lr=final_value,
        verbose=False)

    values = []
    for _ in range(iters):
        values.append(scheduler.get_lr())
        scheduler.step()

    schedule = np.concatenate((warmup_schedule, values))

    assert len(schedule) == total_iters, ''
    return schedule


def multistep_scheduler(base_value,
                        epochs,
                        niter_per_epoch,
                        milestones,
                        gamma=0.1,
                        final_value=0,
                        total_iter=0,
                        start_warmup_value=0,
                        warmup_iters=0):
    '''polynomial_scheduler
    '''

    warmup_schedule = np.array([])
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    # iters = epochs * niter_per_epoch - warmup_iters

    scheduler = paddle.optimizer.lr.MultiStepDecay(
        base_value, milestones, gamma=gamma, last_epoch=-1, verbose=False)

    # scheduler = paddle.optimizer.lr.PiecewiseDecay(
    #     learning_rate=base_value,
    #     decay_steps=iters,
    #     end_lr=final_value,
    #     verbose=False)

    values = []
    for _ in range(epochs):
        _v = scheduler.get_lr()
        _values = [_v for _ in range(niter_per_epoch)]
        scheduler.step()

        values.extend(_values)

    for i in range(warmup_iters):
        values[i] = warmup_schedule[i]

    schedule = values

    # schedule = np.concatenate((warmup_schedule, values))

    print(len(schedule), epochs * niter_per_epoch)

    assert len(schedule) == epochs * niter_per_epoch, ''
    return schedule


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0,
                     warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)

    schedule = np.array([
        final_value + 0.5 * (base_value - final_value) *
        (1 + math.cos(math.pi * i / (len(iters)))) for i in iters
    ])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


# optimizer = create_optimizer(model, skip_list=skip_weight_decay_list, num_layers=num_layers, decay_dict=decay_dict)  
# lr_scheduler_values = polynomial_scheduler(base_value=1e-4, final_value=0., total_iters=iters, start_warmup_value=0, warmup_iters=1500) 
# optimizer.set_lr(lr_scheduler_values[iter])


def create_optimizer(
        model,
        filter_bias_and_bn=True,
        num_layers=None,
        skip_decay_list=None,
        decay_dict=None,
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        layer_decay=0.65, ):
    # opt_lower = args.opt.lower()
    # opt_lower = 'adamw'

    # weight_decay = weight_decay

    if weight_decay and filter_bias_and_bn:
        # skip = {}
        # if skip_decay_list is not None:
        #     skip = skip_decay_list
        # elif hasattr(model, 'no_weight_decay'):
        #     skip = model.no_weight_decay()

        # decay_dict = {
        #     param.name: not (len(param.shape) == 1 or name.endswith(".bias") or
        #                      name in skip_decay_list)
        #     for name, param in model.named_parameters()
        # }

        # decay_dict = {
        #     param.name: not (len(param.shape) == 1 or name.endswith(".bias") or
        #                      #  name in skip_decay_list)
        #                      any([_n in name for _n in skip_decay_list]))
        #     for name, param in model.named_parameters()
        # }

        # TODO same as pytorch, cls_token weight_decay=True
        decay_dict = {
            param.name: not (len(param.shape) == 1 or name.endswith(".bias") or
                             any([_n in name for _n in ['pos_embed']]))
            for name, param in model.named_parameters()
        }

        parameters = [param for param in model.parameters()]

        # TODO
        # weight_decay = 0.

        _decay_dict = {
            name: not (len(param.shape) == 1 or name.endswith(".bias") or any(
                [_n in name for _n in ['pos_embed']]))
            for name, param in model.named_parameters()
        }
        print(_decay_dict)
        del _decay_dict

    else:
        parameters = model.parameters()

    # opt_args = dict(learning_rate=args.lr, weight_decay=weight_decay)
    opt_args = dict(learning_rate=lr, weight_decay=weight_decay)

    opt_args['parameters'] = parameters
    if decay_dict is not None:
        opt_args['apply_decay_param_fun'] = lambda n: decay_dict[n]

    # if hasattr(args, 'opt_eps') and args.opt_eps is not None:
    #     opt_args['epsilon'] = args.opt_eps
    # if hasattr(args, 'opt_betas') and args.opt_betas is not None:
    #     opt_args['beta1'] = args.opt_betas[0]
    #     opt_args['beta2'] = args.opt_betas[1]

    opt_args['beta1'] = betas[0]
    opt_args['beta2'] = betas[1]

    # layer_decay = layer_decay

    opt_args['layerwise_decay'] = layer_decay
    name_dict = dict()
    for n, p in model.named_parameters():
        name_dict[p.name] = n

    opt_args['name_dict'] = name_dict
    opt_args['n_layers'] = num_layers

    # if hasattr(args, 'layer_decay') and args.layer_decay < 1.0:
    #     opt_args['layerwise_decay'] = args.layer_decay
    #     name_dict = dict()
    #     for n, p in model.named_parameters():
    #         name_dict[p.name] = n
    #     opt_args['name_dict'] = name_dict 
    #     opt_args['n_layers'] = num_layers 

    # opt_split = opt_lower.split('_')
    # opt_lower = opt_split[-1]

    optimizer = AdamWDL(**opt_args)

    return optimizer
