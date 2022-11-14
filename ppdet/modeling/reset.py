import paddle
import paddle.nn as nn


def reset_model(model, old_type, new_type=None, reset_func=None, **kwargs):
    def default_reset_func(module, new_type=new_type):
        return new_type()

    reset_func = default_reset_func if reset_func is None else reset_func

    if isinstance(model, old_type):
        model = reset_func(model, **kwargs)
    else:
        for name, child in model.named_children():
            _child = reset_model(child, old_type, new_type, reset_func,
                                 **kwargs)
            if _child is not child:
                setattr(model, name, _child)

    return model


def replace_bn_gn(m, num_groups=8):
    '''replace_bn_gn
    '''
    assert isinstance(m, nn.BatchNorm2D), ''
    _m = nn.GroupNorm(
        num_groups,
        m._num_features,
        epsilon=1e-6,
        weight_attr=m._weight_attr,
        bias_attr=m._bias_attr)

    return _m


if __name__ == '__main__':
    pass

    m = nn.Sequential(
        nn.Conv2D(
            3,
            2,
            1, ),
        nn.BatchNorm2D(10),
        nn.Sequential(
            nn.Conv2D(
                3,
                2,
                1, ),
            nn.BatchNorm2D(20), ), nn.ReLU())

    _m = reset_model(m, nn.BatchNorm2D, nn.GroupNorm, reset_func=replace_bn_gn)

    print(m)
