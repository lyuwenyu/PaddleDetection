import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention, _convert_attention_mask
from .position_encoding import PositionEmbedding
from .utils import _get_clones
from ..initializer import linear_init_, conv_init_, xavier_uniform_, normal_


class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout3 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                pos_embed=None,
                query_pos_embed=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        q = self.with_pos_embed(tgt, query_pos_embed)
        k = self.with_pos_embed(memory, pos_embed)
        tgt = self.cross_attn(q, k, value=memory, attn_mask=memory_mask)
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Layer):
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                pos_embed=None,
                query_pos_embed=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                pos_embed=pos_embed,
                query_pos_embed=query_pos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output.unsqueeze(0)


class TDeocder(nn.Layer):
    def __init__(self, ):

        hidden_dim = 1024
        nhead = hidden_dim // 64
        dropout = 0
        activation = 'gelu'
        normalize_before = False
        dim_feedforward = 2048
        attn_dropout = None
        act_dropout = None
        num_decoder_layers = 3
        return_intermediate_dec = False

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)

    def forward(self, memory):

        query_pos_embed = self.query_pos_embed.weight.unsqueeze(0).tile(
            [bs, 1, 1])
        tgt = paddle.zeros_like(query_pos_embed)
        output = self.decoder(
            tgt,
            memory,
            memory_mask=src_mask,
            pos_embed=pos_embed,
            query_pos_embed=query_pos_embed)

        return (output, memory.transpose([0, 2, 1]).reshape([bs, c, h, w]),
                src_proj, src_mask.reshape([bs, 1, 1, h, w]))
