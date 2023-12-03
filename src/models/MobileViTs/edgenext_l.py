from enum import StrEnum, auto
import torch
from torch import nn
from timm.models.layers import trunc_normal_
from .layers import LayerNorm, PositionalEncodingFourier, LandmarksLayer
from .sdta_encoder import SDTAEncoder
from .conv_encoder import ConvEncoder

class EdgeNeXt_Models(StrEnum):
    Base = "edgenext_base"
    Small = "edgenext_small"
    Spatial_Attention_Small = "edgenext_custom_a"





'''
Standart EdgeNeXt Base with Dense final layer
'''
def edgenext_l_base(num_landmarks):
    # 18.51M & 3840.93M @ 256 resolution
    # 82.5% (normal) 83.7% (USI) Top-1 accuracy
    # AA=True, Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=xx.xx versus xx.xx for MobileViT_S
    # For A100: FPS @ BS=1: xxx.xx & @ BS=256: xxxx.xx
    model = EdgeNeXt(num_landmarks=num_landmarks, num_blocks_stage=[3, 3, 9, 3], 
                     depth_stage=[80, 160, 288, 584], expan_ratio=4,
                     num_global_blocks_stage=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_stage_positional_encoding=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     **{"classifier_dropout":0})

    return model

'''
Standart EdgeNeXt Small with Dense final layer
'''
def edgenext_l_small(num_landmarks):
    # 5.59M & 1260.59M @ 256 resolution
    # 79.43% Top-1 accuracy
    # AA=True, No Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=20.47 versus 18.86 for MobileViT_S
    # For A100: FPS @ BS=1: 172.33 & @ BS=256: 3010.25 versus FPS @ BS=1: 93.84 & @ BS=256: 1785.92 for MobileViT_S
    model = EdgeNeXt(num_landmarks=num_landmarks, num_blocks_stage=[3, 3, 9, 3], 
                     depth_stage=[48, 96, 160, 304], expan_ratio=4,
                     num_global_blocks_stage=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_stage_positional_encoding=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     **{"classifier_dropout":0})

    return model


'''
--------------------------------------------
ABLATIONSSSSS ------------------------------
--------------------------------------------
'''

'''
Modified from edgenext_small (a.)
(170ms, 33ms)
- XCA for XSA
'''
def edgenext_l_custom_a(num_landmarks=98):
    model = EdgeNeXt(num_landmarks=num_landmarks, num_blocks_stage=[3, 3, 9, 3], 
                     depth_stage=[48, 96, 160, 304], expan_ratio=4,
                     num_global_blocks_stage=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_stage_positional_encoding=[False, True, False, False],
                     use_spatial_attention=[True,True,True,True],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     **{"classifier_dropout":0})
    return model


'''
Modified from edgenext_small (b.)
(170ms, 31ms)
- Depths [3,4,5,6] instead of [3,3,9,3]
- d2_scales [2,3,4,5] instead of [2,2,3,4]
'''
def edgenext_l_custom_b(num_landmarks):
    model = EdgeNeXt(num_landmarks=num_landmarks, num_blocks_stage=[3, 4, 5, 6], 
                     depth_stage=[48, 96, 160, 304], expan_ratio=4,
                     num_global_blocks_stage=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_stage_positional_encoding=[False, True, False, False],
                     use_spatial_attention=[False,False,False,False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model


'''
Modified from edgenext_small (c.)
(170ms, 33ms)
- XCA for XSA
- Depths [3,4,5,6] instead of [3,3,9,3]
- d2_scales [2,3,4,5] instead of [2,2,3,4]
'''
def edgenext_l_custom_c(num_landmarks=98):
    model = EdgeNeXt(num_landmarks=num_landmarks, num_blocks_stage=[3, 4, 5, 6], 
                     depth_stage=[48, 96, 160, 304], expan_ratio=4,
                     num_global_blocks_stage=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_stage_positional_encoding=[False, True, False, False],
                     use_spatial_attention=[True,True,True,True],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model


'''
(d.) 
(70ms, 31ms)
- Reduce first stage dim to 24 from 48 (half)
- 2,3,2,1 Attention blocks
- 1 Convolution / Stage
'''
def edgenext_l_custom_d(num_landmarks=98, spatial_attention=False):
    model = EdgeNeXt(num_landmarks=num_landmarks, num_blocks_stage=[3, 4, 3, 2], 
                     depth_stage=[24, 96, 160, 304], expan_ratio=4,
                     num_global_blocks_stage=[2, 3, 2, 1],
                     global_block_type=['SDTA', 'SDTA', 'SDTA', 'SDTA'],
                     use_stage_positional_encoding=[False, True, False, False],
                     use_spatial_attention=[spatial_attention, spatial_attention, spatial_attention, spatial_attention],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model

'''
(e.) (135ms, 41ms)
- More STDA blocks!! 
- 5, 4, 2, 1 Attention blocks
- 1 Conv block first 2 stages + 2 Conv block / Last 2 stages
'''
def edgenext_l_custom_e(num_landmarks=98, spatial_attention=False):
    model = EdgeNeXt(num_landmarks=num_landmarks, num_blocks_stage=[6, 5, 4, 3], 
                     depth_stage=[48, 96, 160, 304], expan_ratio=4,
                     num_global_blocks_stage=[5, 4, 2, 1],
                     global_block_type=['SDTA', 'SDTA', 'SDTA', 'SDTA'],
                     use_stage_positional_encoding=[False, False, False, False],
                     use_spatial_attention=[spatial_attention, spatial_attention, spatial_attention, spatial_attention],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model

'''
Same as (e.) but no convolutions 
(80ms 23ms)
'''
def edgenext_l_custom_f(num_landmarks=98, spatial_attention=False):
    model = EdgeNeXt(num_landmarks=num_landmarks, num_blocks_stage=[5, 4, 2, 1], 
                     depth_stage=[24, 96, 160, 304], expan_ratio=4,
                     num_global_blocks_stage=[5, 4, 2, 1],
                     global_block_type=['SDTA', 'SDTA', 'SDTA', 'SDTA'],
                     use_stage_positional_encoding=[False, False, False, False],
                     use_spatial_attention=[spatial_attention, spatial_attention, spatial_attention, spatial_attention],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model


'''
EdgeNeXt Architecture

in_chans:                 Num of channels of input img
num_classes:              Num of prediction head outputs
depth_stage:              Num of blocks on each of the stages
num_global_blocks_stage:  Ammount of Global blocks to use in each stage (STDA)
depth_stage:              Channels of the image on each stage
num_attention_heads:      Number of heads for MHSA (in STDA) in each stage
'''
class EdgeNeXt(nn.Module):
    def __init__(self, in_chans=3, num_landmarks=98,
                 num_blocks_stage=[3, 3, 9, 3], 
                 depth_stage=[24, 48, 88, 168],
                 num_global_blocks_stage=[0, 0, 0, 3],
                 drop_path_rate=0., 
                 layer_scale_init_value=1e-6, 
                 expan_ratio=4,
                 kernel_sizes=[7, 7, 7, 7], 
                 num_attention_heads=[8, 8, 8, 8], 
                 use_stage_positional_encoding=[False, False, False, False], 
                 use_global_positional_encoding=False,
                 use_spatial_attention=[False, False, False, False],
                 d2_scales=[2, 3, 4, 5], **kwargs):
        super().__init__()
        
        if use_global_positional_encoding:
            self.pos_embd = PositionalEncodingFourier(dim=depth_stage[0])
        else:
            self.pos_embd = None
        
        # Downsampling conv layers
        self.downsample_layers = nn.ModuleList()  
        stem = nn.Sequential(
            nn.Conv2d(in_chans, depth_stage[0], kernel_size=4, stride=4), # 4x4
            LayerNorm(depth_stage[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        downsample_layer1 = nn.Sequential(
            LayerNorm(depth_stage[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(depth_stage[0], depth_stage[1], kernel_size=2, stride=2), # 2x2
        )
        self.downsample_layers.append(downsample_layer1)
        downsample_layer2 = nn.Sequential(
            LayerNorm(depth_stage[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(depth_stage[1], depth_stage[2], kernel_size=2, stride=2),
        )
        self.downsample_layers.append(downsample_layer2)
        downsample_layer3 = nn.Sequential(
            LayerNorm(depth_stage[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(depth_stage[2], depth_stage[3], kernel_size=2, stride=2),
        )
        self.downsample_layers.append(downsample_layer3)

        self.stages = nn.ModuleList()  
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks_stage))]
        cur = 0
        for stage, n_blocks_in_stage in enumerate(num_blocks_stage):
            stage_blocks = []
            for current_block in range(n_blocks_in_stage):
                if current_block > n_blocks_in_stage - num_global_blocks_stage[stage] - 1:
                    stage_blocks.append(SDTAEncoder(dim=depth_stage[stage], drop_path=dp_rates[cur + current_block],
                                                    expan_ratio=expan_ratio, scales=d2_scales[stage],
                                                    use_pos_emb=use_stage_positional_encoding[stage], num_heads=num_attention_heads[stage], 
                                                    use_spatial_attention=use_spatial_attention[stage]))                    
                else:
                    stage_blocks.append(ConvEncoder(dim=depth_stage[stage], drop_path=dp_rates[cur + current_block],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio, kernel_size=kernel_sizes[stage]))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += n_blocks_in_stage

        self.norm = nn.LayerNorm(depth_stage[-1], eps=1e-6)  # Final norm layer
        
        '''
        HEAD!!
        '''
        inhead = depth_stage[-1] 
        self.head = LandmarksLayer(in_features=inhead, num_landmarks=num_landmarks)

        self.apply(self._init_weights)
        self.head_dropout = nn.Dropout(kwargs["classifier_dropout"])


    def _init_weights(self, m):  
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd:
            B, C, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x) 

        return self.norm(x.mean([-2, -1]))  # Global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.head_dropout(x))

        return x
