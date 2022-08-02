import paddle
import paddle.nn as nn
from ppdet.modeling.bbox_utils import bbox_iou


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class IouLoss(object):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(self,
                 loss_weight=2.5,
                 giou=False,
                 diou=False,
                 ciou=False,
                 loss_square=True):
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou
        self.loss_square = loss_square

    def __call__(self, pbox, gbox):
        iou = bbox_iou(
            pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        if self.loss_square:
            loss_iou = 1 - iou * iou
        else:
            loss_iou = 1 - iou

        loss_iou = loss_iou * self.loss_weight
        return loss_iou


from paddle.fluid.data_feeder import convert_dtype


def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.

    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.

    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


class TransformerEncoder(nn.Layer):
    """
    TransformerEncoder is a stack of N encoder layers. 

    Parameters:
        encoder_layer (Layer): an instance of the `TransformerEncoderLayer`. It
            would be used as the first layer, and the other layers would be created
            according to the configurations of it.
        num_layers (int): The number of encoder layers to be stacked.
        norm (LayerNorm, optional): the layer normalization component. If provided,
            apply layer normalization on the output of last encoder layer.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.nn import TransformerEncoderLayer, TransformerEncoder

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, n_head, src_len, src_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            encoder_layer = TransformerEncoderLayer(128, 2, 512)
            encoder = TransformerEncoder(encoder_layer, 2)
            enc_output = encoder(enc_input, attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 encoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.LayerList([(
            encoder_layer
            if i == 0 else type(encoder_layer)(**encoder_layer._config))
                                    for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src, src_mask=None, cache=None):
        r"""
        Applies a stack of N Transformer encoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last encoder
        layer.

        Parameters:
            src (Tensor): The input of Transformer encoder. It is a tensor
                with shape `[batch_size, sequence_length, d_model]`. The data
                type should be float32 or float64.
            src_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
            cache (list, optional): It is a list, and each element in the list
                is `incremental_cache` produced by `TransformerEncoderLayer.gen_cache`. 
                See `TransformerEncoder.gen_cache` for more details. It is only
                used for inference and should be None for training. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `src`, representing the output of Transformer encoder. \
                Or a tuple if `cache` is not None, except for encoder output, \
                the tuple includes the new cache which is same as input `cache` \
                argument but `incremental_cache` in it has an incremental length. \
                See `MultiHeadAttention.gen_cache` and `MultiHeadAttention.forward` \
                for more details.
        """
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        output = src
        new_caches = []
        outputs = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output, src_mask=src_mask)
            else:
                output, new_cache = mod(output,
                                        src_mask=src_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

            outputs.append(output)

        # print(self.norm) 
        if self.norm is not None:
            output = self.norm(output)

        if not self.return_intermediate:
            return output if cache is None else (output, new_caches)
        else:
            return outputs

    def gen_cache(self, src):
        r"""
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is `incremental_cache` produced by 
        `TransformerEncoderLayer.gen_cache`. See `TransformerEncoderLayer.gen_cache`
        for more details.

        Parameters:
            src (Tensor): The input of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.

        Returns:
            list: It is a list, and each element in the list is `incremental_cache` 
            produced by `TransformerEncoderLayer.gen_cache`. See 
            `TransformerEncoderLayer.gen_cache` for more details.
        """
        cache = [layer.gen_cache(src) for layer in self.layers]
        return cache
