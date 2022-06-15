# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle.nn.initializer import Constant
from paddle.distributed.fleet.utils import recompute

# from paddleseg.cvlibs import manager
# from paddleseg.utils import utils, logger

# from ppdet.modeling.backbones.transformer_utils import to_2tuple, DropPath, Identity
from .transformer_utils import to_2tuple, DropPath, Identity

from ppdet.modeling.shape_spec import ShapeSpec
from ppdet.core.workspace import register, serializable

zeros_ = Constant(value=0.)

global sync_bn_feat
sync_bn_feat = None

# for checkpoint
paddle.seed(0)


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 window_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        ###########add by xy
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=False)

        if qkv_bias:
            self.q_bias = self.create_parameter(
                shape=([dim]), default_initializer=zeros_)
            self.v_bias = self.create_parameter(
                shape=([dim]), default_initializer=zeros_)
            #self.q_bias = nn.Parameter(paddle.zeros(dim)) 
            #self.v_bias = nn.Parameter(paddle.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = self.create_parameter(
                shape=(self.num_relative_distance, num_heads),
                default_initializer=zeros_)  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = paddle.arange(window_size[0])
            coords_w = paddle.arange(window_size[1])
            coords = paddle.stack(paddle.meshgrid(
                [coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww 
            coords_flatten_1 = paddle.unsqueeze(coords_flatten, 2)
            coords_flatten_2 = paddle.unsqueeze(coords_flatten, 1)
            relative_coords = coords_flatten_1.clone() - coords_flatten_2.clone(
            )

            #relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Wh
            relative_coords = relative_coords.transpose(
                (1, 2, 0))  #.contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[
                0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                paddle.zeros(shape=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(
                -1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index",
                                 relative_position_index)
            # trunc_normal_(self.relative_position_bias_table, std=.0)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        ############add by xy done

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        x_shape = paddle.shape(x)
        N, C = x_shape[1], x_shape[2]

        ###### add by xy
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = paddle.concat(
                (self.q_bias, paddle.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)

        qkv = qkv.reshape((-1, N, 3, self.num_heads,
                           C // self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.reshape([-1])].reshape([
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1
                ])  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose(
                (2, 0, 1))  #.contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 window_size=None,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5):
        super().__init__()
        # self.norm1 = eval(norm_layer)(dim, epsilon=1e-6)
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        # add by xy
        if init_values is not None:
            self.gamma_1 = self.create_parameter(
                shape=([dim]), default_initializer=Constant(value=init_values))
            self.gamma_2 = self.create_parameter(
                shape=([dim]), default_initializer=Constant(value=init_values))
            #self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            #self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):

        # print('x', x, x.mean(), x.sum())
        # print('rel_pos_bias', rel_pos_bias, )
        # print('self.norm1(x)', self.norm1(x))
        # print('self.norm1.weight', self.norm1.weight)
        # print('self.norm1.bias', self.norm1.bias)

        # y = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
        # print('y', y)

        if self.gamma_1 is None:
            x = x + self.drop_path(
                self.attn(
                    self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    @property
    def num_patches_in_h(self):
        return self.img_size[1] // self.patch_size[1]

    @property
    def num_patches_in_w(self):
        return self.img_size[0] // self.patch_size[0]

    @property
    def patch_shape(self):
        return (self.img_size[0] // self.patch_size[0],
                self.img_size[1] // self.patch_size[1])

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=[224, 224],
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        self.num_patches_w = img_size[0] // patch_size
        self.num_patches_h = img_size[1] // patch_size

        num_patches = self.num_patches_w * self.num_patches_h
        self.patch_shape = (img_size[0] // patch_size,
                            img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        print("##############patch here!!!")
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    @property
    def num_patches_in_h(self):
        return self.img_size[1] // self.patch_size

    @property
    def num_patches_in_w(self):
        return self.img_size[0] // self.patch_size

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        return self.proj(x)


###############add by xy
class RelativePositionBias(nn.Layer):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = self.create_parameter(
            shape=(self.num_relative_distance, num_heads),
            default_initialize=zeros_)
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(window_size[0])
        coords_w = paddle.arange(window_size[1])
        coords = paddle.stack(paddle.meshgrid(
            [coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = coords.flatten(1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpos(
            (1, 2, 0))  #.contiguous()  # Wh*Ww, Wh*Ww, 2 
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            paddle.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(
            -1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer("relative_position_index", relative_position_index)
        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                 self.window_size[0] * self.window_size[1] + 1,
                 self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH 
        return relative_position_bias.transpose(
            (2, 0, 1))  #.contiguous()  # nH, Wh*Ww, Wh*Ww


def get_sinusoid_encoding_table(n_position, d_hid, token=False):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    if token:
        sinusoid_table = np.concatenate(
            [sinusoid_table, np.zeros([1, d_hid])], dim=0)
    # return Tensor(sinusoid_table, dtype=float32).unsqueeze(0)
    return paddle.to_tensor(sinusoid_table, dtype=paddle.float32).unsqueeze(0)


#############add by xy done

# @manager.BACKBONES.add_component


@register
@serializable
class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch input
    """

    def __init__(self,
                 img_size=[672, 1092],
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 init_values=None,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 epsilon=1e-5,
                 final_norm=False,
                 pretrained=None,
                 out_indices=[3, 5, 7, 11],
                 use_abs_pos_emb=False,
                 use_sincos_pos_emb=True,
                 use_checkpoint=False,
                 **args):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)

        self.pos_w = self.patch_embed.num_patches_in_w
        self.pos_h = self.patch_embed.num_patches_in_h

        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))

        if use_abs_pos_emb:
            self.pos_embed = self.create_parameter(
                shape=(1, self.pos_w * self.pos_h + 1, embed_dim),
                default_initializer=paddle.nn.initializer.TruncatedNormal(
                    std=.02))
        elif use_sincos_pos_emb:
            pos_embed = self.build_2d_sincos_position_embedding(embed_dim)
            self.pos_embed = pos_embed

            self.pos_embed = self.create_parameter(shape=pos_embed.shape)
            self.pos_embed.set_value(pos_embed.numpy())
            self.pos_embed.stop_gradient = True

        else:
            self.pos_embed = None

        # self.pos_embed = self.create_parameter(
        #     shape=(1, self.pos_w * self.pos_h + 1, embed_dim),
        #     default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        self.pos_drop = nn.Dropout(p=drop_rate)

        #################add by xy
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        # elif self.use_sincos_pos_emb:
        #     self.pos_embed = self.build_2d_sincos_position_embedding(embed_dim)

        else:
            self.rel_pos_bias = None

        self.use_rel_pos_bias = use_rel_pos_bias

        dpr = np.linspace(0, drop_path_rate, depth)

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.patch_shape
                if use_rel_pos_bias else None,
                epsilon=epsilon) for i in range(depth)
        ])

        self.final_norm = final_norm
        ######### del by xy
        #if self.final_norm:
        #    self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)
        self.pretrained = pretrained
        self.init_weight()

        assert len(out_indices) <= 4, ''
        self.out_indices = out_indices
        self.out_channels = [embed_dim for _ in range(len(out_indices))]
        self.out_strides = [4, 8, 16, 32][-len(out_indices):]

        self.norm = Identity()
        self.init_fpn(
            embed_dim=embed_dim,
            patch_size=patch_size, )

    def init_weight(self):
        # utils.load_pretrained_model(self, self.pretrained)

        # # load and resize pos_embed
        # model_path = self.pretrained
        # if not os.path.exists(model_path):
        #     model_path = utils.download_pretrained_model(model_path)

        pretrained = self.pretrained

        if pretrained:
            if 'http' in pretrained:  #URL
                path = paddle.utils.download.get_weights_path_from_url(
                    pretrained)
            else:  #model in local path
                path = pretrained
            # self.set_state_dict(paddle.load(path))

            load_state_dict = paddle.load(path)
            model_state_dict = self.state_dict()
            pos_embed_name = "pos_embed"
            if pos_embed_name in load_state_dict.keys():
                load_pos_embed = paddle.to_tensor(
                    load_state_dict[pos_embed_name], dtype="float32")
                if self.pos_embed.shape != load_pos_embed.shape:
                    pos_size = int(math.sqrt(load_pos_embed.shape[1] - 1))
                    model_state_dict[pos_embed_name] = self.resize_pos_embed(
                        load_pos_embed, (pos_size, pos_size),
                        (self.pos_h, self.pos_w))

                    # self.set_state_dict(model_state_dict)
                    load_state_dict[pos_embed_name] = model_state_dict[
                        pos_embed_name]

                    print("Load pos_embed and resize it from {} to {} .".format(
                        load_pos_embed.shape, self.pos_embed.shape))

                    # logger.info(
                    #     "Load pos_embed and resize it from {} to {} .".format(
                    #         load_pos_embed.shape, self.pos_embed.shape))

            self.set_state_dict(load_state_dict)
            print("Load load_state_dict....")

    def init_fpn(self, embed_dim=768, patch_size=16, out_with_norm=False):
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2),

                # nn.SyncBatchNorm(
                #     embed_dim, momentum=0.1),
                nn.BatchNorm2D(embed_dim),
                nn.GELU(),
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2), )

            self.fpn2 = nn.Sequential(
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2), )

            self.fpn3 = Identity()

            self.fpn4 = nn.MaxPool2D(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2), )

            self.fpn2 = Identity()

            self.fpn3 = nn.Sequential(nn.MaxPool2D(kernel_size=2, stride=2), )

            self.fpn4 = nn.Sequential(nn.MaxPool2D(kernel_size=4, stride=4), )

        #########add by xy
        if not out_with_norm:
            self.norm = Identity()
        else:
            self.norm = nn.LayerNorm(embed_dim, epsilon=1e-6)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        if npatch == N and w0 == self.patch_embed.num_patches_w and h0 == self.patch_embed.num_patches_h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        # tmp = patch_pos_embed.reshape([
        #     1, self.patch_embed.num_patches_w, self.patch_embed.num_patches_h,
        #     dim
        # ]).transpose((0, 3, 1, 2))

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape([
                1, self.patch_embed.num_patches_w,
                self.patch_embed.num_patches_h, dim
            ]).transpose((0, 3, 1, 2)),
            scale_factor=(w0 / self.patch_embed.num_patches_w,
                          h0 / self.patch_embed.num_patches_h),
            mode='bicubic', )
        assert int(w0) == patch_pos_embed.shape[-2] and int(
            h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.transpose(
            (0, 2, 3, 1)).reshape([1, -1, dim])
        return paddle.concat(
            (class_pos_embed.unsqueeze(0), patch_pos_embed), axis=1)

    def resize_pos_embed(self, pos_embed, old_hw, new_hw):
        """
        Resize pos_embed weight.
        Args:
            pos_embed (Tensor): the pos_embed weight
            old_hw (list[int]): the height and width of old pos_embed
            new_hw (list[int]): the height and width of new pos_embed
        Returns:
            Tensor: the resized pos_embed weight
        """
        cls_pos_embed = pos_embed[:, :1, :]
        pos_embed = pos_embed[:, 1:, :]

        pos_embed = pos_embed.transpose([0, 2, 1])
        pos_embed = pos_embed.reshape([1, -1, old_hw[0], old_hw[1]])
        pos_embed = F.interpolate(
            pos_embed, new_hw, mode='bicubic', align_corners=False)
        pos_embed = pos_embed.flatten(2).transpose([0, 2, 1])
        pos_embed = paddle.concat([cls_pos_embed, pos_embed], axis=1)

        return pos_embed

    def build_2d_sincos_position_embedding(
            self,
            embed_dim=768,
            temperature=10000., ):
        h, w = self.patch_embed.patch_shape
        grid_w = paddle.arange(w, dtype=paddle.float32)
        grid_h = paddle.arange(h, dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        # out_w = paddle.einsum('m,d->md', [grid_w.flatten(), omega])
        # out_h = paddle.einsum('m,d->md', [grid_h.flatten(), omega])

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        pos_emb = paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

        pe_token = paddle.zeros([1, 1, embed_dim], dtype=paddle.float32)
        pos_embed = paddle.concat([pe_token, pos_emb], axis=1)
        # pos_embed.stop_gradient = True

        return pos_embed

    def _build_2d_sincos_position_embedding(self,
                                            embed_dim=768,
                                            temperature=10000.,
                                            decode=False):
        h, w = self.patch_embed.patch_shape
        grid_w = paddle.arange(w, dtype=paddle.float32)
        grid_h = paddle.arange(h, dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = paddle.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = paddle.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = paddle.cat([
            paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
            paddle.cos(out_h)
        ],
                             dim=1)[None, :, :]

        pe_token = paddle.zeros([1, 1, embed_dim], dtype=paddle.float32)
        pos_embed = nn.Parameter(paddle.cat([pe_token, pos_emb], dim=1))
        pos_embed.stop_gradient = True
        return pos_embed

    def forward(self, x):
        x = x['image'] if isinstance(x, dict) else x
        _, _, w, h = x.shape

        x = self.patch_embed(x)

        # print('patch_embed', x)

        x_shape = paddle.shape(x)  # b * c * h * w

        cls_tokens = self.cls_token.expand((x_shape[0], -1, -1))
        x = x.flatten(2).transpose([0, 2, 1])  # b * hw * c
        x = paddle.concat([cls_tokens, x], axis=1)

        if self.pos_embed is not None:
            x = x + self.interpolate_pos_encoding(x, w, h)

        #         if paddle.shape(x)[1] == self.pos_embed.shape[1]:
        #             x = x + self.pos_embed
        #         else:
        #             x = x + self.resize_pos_embed(self.pos_embed,
        #                                           (self.pos_h, self.pos_w), x_shape[2:])

        # print('pos_embed', x)

        x = self.pos_drop(x)

        #########add by xy
        rel_pos_bias = self.rel_pos_bias(
        ) if self.rel_pos_bias is not None else None

        B, _, Hp, Wp = x_shape

        # print(x)

        feats = []
        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = recompute(blk, x, rel_pos_bias,
                              **{"preserve_rng_state": True})
            else:
                x = blk(x, rel_pos_bias)

            ###########del by xy
            #if self.final_norm and idx == len(self.blocks) - 1:
            #    x = self.norm(x)

            # feats.append(x[:, 1:, :])

            # print(x, )
            # ccc+=1

            if idx in self.out_indices:
                xp = paddle.reshape(
                    paddle.transpose(
                        self.norm(x[:, 1:, :]), perm=[0, 2, 1]),
                    shape=[B, -1, Hp, Wp])
                feats.append(xp)

            #         # TODO make sure `norm` here is right ?
            #         B, _, Hp, Wp = x_shape
            #         feats = [feats[i] for i in self.out_indices]
            #         for i, feat in enumerate(feats):
            #             feats[i] = paddle.reshape(
            #                 paddle.transpose(
            #                     self.norm(feat), perm=[0, 2, 1]),
            #                 shape=[B, -1, Hp, Wp])

            # TODO make sure feats convert to fpn input format
        fpns = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(feats)):
            feats[i] = fpns[i](feats[i])

            if i == 0:
                global sync_bn_feat
                sync_bn_feat = feats[0]

        return feats

    @property
    def num_layers(self):
        return len(self.blocks)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=c, stride=s)
            for c, s in zip(self.out_channels, self.out_strides)
        ]


# {'vision_transformer_0.w_0': False, 'vision_transformer_0.w_1': False, 'conv2d_0.w_0': True, 'conv2d_0.b_0': False, 'block_0.w_0': False, 'block_0.w_1': False, 'layer_norm_0.w_0': False, 'layer_norm_0.b_0': False, 'attention_0.w_0': False, 'attention_0.w_1': False, 'linear_0.w_0': True, 'linear_1.w_0': True, 'linear_1.b_0': False, 'layer_norm_1.w_0': False, 'layer_norm_1.b_0': False, 'linear_2.w_0': True, 'linear_2.b_0': False, 'linear_3.w_0': True, 'linear_3.b_0': False, 'block_1.w_0':False, 'block_1.w_1': False, 'layer_norm_2.w_0': False, 'layer_norm_2.b_0': False, 'attention_1.w_0': False, 'attention_1.w_1': False, 'linear_4.w_0': True, 'linear_5.w_0': True, 'linear_5.b_0': False, 'layer_norm_3.w_0': False, 'layer_norm_3.b_0': False, 'linear_6.w_0': True, 'linear_6.b_0': False, 'linear_7.w_0': True, 'linear_7.b_0': False, 'block_2.w_0': False, 'block_2.w_1': False, 'layer_norm_4.w_0': False, 'layer_norm_4.b_0': False, 'attention_2.w_0': False, 'attention_2.w_1': False, 'linear_8.w_0': True, 'linear_9.w_0': True, 'linear_9.b_0': False, 'layer_norm_5.w_0': False, 'layer_norm_5.b_0': False, 'linear_10.w_0': True, 'linear_10.b_0': False, 'linear_11.w_0': True, 'linear_11.b_0': False, 'block_3.w_0': False, 'block_3.w_1': False, 'layer_norm_6.w_0': False, 'layer_norm_6.b_0': False, 'attention_3.w_0': False, 'attention_3.w_1': False,'linear_12.w_0': True, 'linear_13.w_0': True, 'linear_13.b_0': False, 'layer_norm_7.w_0': False, 'layer_norm_7.b_0': False, 'linear_14.w_0': True, 'linear_14.b_0': False, 'linear_15.w_0': True, 'linear_15.b_0': False, 'block_4.w_0': False, 'block_4.w_1': False, 'layer_norm_8.w_0': False, 'layer_norm_8.b_0': False, 'attention_4.w_0': False, 'attention_4.w_1': False, 'linear_16.w_0': True, 'linear_17.w_0': True, 'linear_17.b_0': False, 'layer_norm_9.w_0': False, 'layer_norm_9.b_0': False,'linear_18.w_0': True, 'linear_18.b_0': False, 'linear_19.w_0': True, 'linear_19.b_0': False, 'block_5.w_0': False, 'block_5.w_1': False, 'layer_norm_10.w_0': False, 'layer_norm_10.b_0': False, 'attention_5.w_0': False, 'attention_5.w_1': False,'linear_20.w_0': True, 'linear_21.w_0': True, 'linear_21.b_0': False, 'layer_norm_11.w_0': False, 'layer_norm_11.b_0': False, 'linear_22.w_0': True, 'linear_22.b_0': False, 'linear_23.w_0': True, 'linear_23.b_0': False, 'block_6.w_0': False, 'block_6.w_1': False, 'layer_norm_12.w_0': False, 'layer_norm_12.b_0': False, 'attention_6.w_0': False, 'attention_6.w_1': False, 'linear_24.w_0': True, 'linear_25.w_0': True, 'linear_25.b_0': False, 'layer_norm_13.w_0': False, 'layer_norm_13.b_0': False, 'linear_26.w_0': True, 'linear_26.b_0': False, 'linear_27.w_0': True, 'linear_27.b_0': False, 'block_7.w_0': False, 'block_7.w_1': False, 'layer_norm_14.w_0': False, 'layer_norm_14.b_0': False, 'attention_7.w_0': False, 'attention_7.w_1': False, 'linear_28.w_0': True, 'linear_29.w_0': True, 'linear_29.b_0': False, 'layer_norm_15.w_0': False, 'layer_norm_15.b_0': False, 'linear_30.w_0': True, 'linear_30.b_0': False, 'linear_31.w_0': True, 'linear_31.b_0': False, 'block_8.w_0': False, 'block_8.w_1': False, 'layer_norm_16.w_0': False, 'layer_norm_16.b_0': False, 'attention_8.w_0': False, 'attention_8.w_1': False, 'linear_32.w_0': True, 'linear_33.w_0': True, 'linear_33.b_0': False, 'layer_norm_17.w_0': False, 'layer_norm_17.b_0': False, 'linear_34.w_0': True, 'linear_34.b_0': False, 'linear_35.w_0': True, 'linear_35.b_0': False, 'block_9.w_0': False, 'block_9.w_1': False, 'layer_norm_18.w_0': False, 'layer_norm_18.b_0': False, 'attention_9.w_0': False, 'attention_9.w_1': False, 'linear_36.w_0': True, 'linear_37.w_0': True, 'linear_37.b_0': False, 'layer_norm_19.w_0': False, 'layer_norm_19.b_0': False, 'linear_38.w_0': True, 'linear_38.b_0': False, 'linear_39.w_0': True, 'linear_39.b_0': False, 'block_10.w_0': False, 'block_10.w_1': False, 'layer_norm_20.w_0': False, 'layer_norm_20.b_0': False, 'attention_10.w_0': False, 'attention_10.w_1': False, 'linear_40.w_0': True, 'linear_41.w_0': True, 'linear_41.b_0': False, 'layer_norm_21.w_0': False, 'layer_norm_21.b_0': False, 'linear_42.w_0': True, 'linear_42.b_0': False, 'linear_43.w_0': True, 'linear_43.b_0': False, 'block_11.w_0': False, 'block_11.w_1': False, 'layer_norm_22.w_0': False, 'layer_norm_22.b_0': False, 'attention_11.w_0': False, 'attention_11.w_1': False, 'linear_44.w_0': True, 'linear_45.w_0': True, 'linear_45.b_0': False, 'layer_norm_23.w_0': False, 'layer_norm_23.b_0': False, 'linear_46.w_0': True, 'linear_46.b_0': False, 'linear_47.w_0': True, 'linear_47.b_0': False,'conv2d_transpose_0.w_0': True, 'conv2d_transpose_0.b_0': False, 'sync_batch_norm_0.w_0': False, 'sync_batch_norm_0.b_0': False, 'sync_batch_norm_0.w_1': False, 'sync_batch_norm_0.w_2': False, 'conv2d_transpose_1.w_0': True, 'conv2d_transpose_1.b_0': False, 'conv2d_transpose_2.w_0': True, 'conv2d_transpose_2.b_0': False, 'conv2d_7.w_0': True, 'conv2d_7.b_0': False, 'conv2d_8.w_0': True, 'conv2d_8.b_0': False, 'conv2d_9.w_0': True, 'conv2d_9.b_0': False, 'conv2d_10.w_0': True, 'batch_norm2d_0.w_0': False, 'batch_norm2d_0.b_0': False, 'batch_norm2d_0.w_1': False, 'batch_norm2d_0.w_2': False, 'conv2d_11.w_0': True, 'batch_norm2d_1.w_0': False, 'batch_norm2d_1.b_0': False, 'batch_norm2d_1.w_1': False, 'batch_norm2d_1.w_2': False, 'conv2d_12.w_0': True, 'batch_norm2d_2.w_0': False, 'batch_norm2d_2.b_0': False, 'batch_norm2d_2.w_1': False, 'batch_norm2d_2.w_2': False, 'conv2d_13.w_0': True, 'batch_norm2d_3.w_0': False, 'batch_norm2d_3.b_0': False, 'batch_norm2d_3.w_1': False, 'batch_norm2d_3.w_2': False, 'linear_48.w_0': True, 'linear_48.b_0': False, 'conv2d_14.w_0': True, 'batch_norm2d_4.w_0': False, 'batch_norm2d_4.b_0': False, 'batch_norm2d_4.w_1': False, 'batch_norm2d_4.w_2': False, 'conv2d_15.w_0': True, 'batch_norm2d_5.w_0': False, 'batch_norm2d_5.b_0': False, 'batch_norm2d_5.w_1': False, 'batch_norm2d_5.w_2': False, 'conv2d_16.w_0': True, 'batch_norm2d_6.w_0': False, 'batch_norm2d_6.b_0': False, 'batch_norm2d_6.w_1': False, 'batch_norm2d_6.w_2': False, 'conv2d_17.w_0': True, 'batch_norm2d_7.w_0': False, 'batch_norm2d_7.b_0': False, 'batch_norm2d_7.w_1': False, 'batch_norm2d_7.w_2': False, 'linear_49.w_0': True, 'linear_49.b_0': False, 'conv2d_18.w_0': True, 'batch_norm2d_8.w_0': False, 'batch_norm2d_8.b_0': False, 'batch_norm2d_8.w_1': False, 'batch_norm2d_8.w_2': False, 'conv2d_19.w_0': True, 'batch_norm2d_9.w_0': False, 'batch_norm2d_9.b_0': False, 'batch_norm2d_9.w_1': False, 'batch_norm2d_9.w_2': False, 'conv2d_20.w_0': True, 'batch_norm2d_10.w_0': False, 'batch_norm2d_10.b_0': False, 'batch_norm2d_10.w_1': False, 'batch_norm2d_10.w_2': False, 'conv2d_21.w_0': True, 'batch_norm2d_11.w_0': False, 'batch_norm2d_11.b_0': False, 'batch_norm2d_11.w_1': False, 'batch_norm2d_11.w_2': False, 'linear_50.w_0': True, 'linear_50.b_0': False, 'linear_51.w_0': True, 'linear_51.b_0': False, 'linear_52.w_0': True, 'linear_52.b_0': False, 'linear_53.w_0': True, 'linear_53.b_0': False, 'linear_54.w_0': True, 'linear_54.b_0': False, 'linear_55.w_0': True, 'linear_55.b_0': False, 'linear_56.w_0': True, 'linear_56.b_0': False, 'conv2d_1.w_0': True, 'conv2d_1.b_0': False, 'conv2d_2.w_0': True, 'conv2d_2.b_0': False, 'conv2d_3.w_0': True, 'conv2d_3.b_0': False, 'conv2d_4.w_0': True, 'conv2d_4.b_0': False, 'conv2d_5.w_0': True, 'conv2d_5.b_0': False, 'conv2d_6.w_0': True, 'conv2d_6.b_0': False}
