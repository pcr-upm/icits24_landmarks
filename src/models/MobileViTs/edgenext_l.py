import torch
from torch import nn
from timm.models.layers import trunc_normal_
from .layers import LayerNorm, PositionalEncodingFourier, LandmarksLayer
from .sdta_encoder import SDTAEncoder
from .conv_encoder import ConvEncoder

'''
model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[80, 160, 288, 584], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     **{"classifier_dropout":0}
                    )
'''
def edgenext_l_base(num_landmarks):
    # 18.51M & 3840.93M @ 256 resolution
    # 82.5% (normal) 83.7% (USI) Top-1 accuracy
    # AA=True, Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=xx.xx versus xx.xx for MobileViT_S
    # For A100: FPS @ BS=1: xxx.xx & @ BS=256: xxxx.xx
    model = EdgeNeXt(num_landmarks=num_landmarks, depths=[3, 3, 9, 3], dims=[80, 160, 288, 584], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     **{"classifier_dropout":0})

    return model

def edgenext_l_small(num_landmarks):
    # 5.59M & 1260.59M @ 256 resolution
    # 79.43% Top-1 accuracy
    # AA=True, No Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=20.47 versus 18.86 for MobileViT_S
    # For A100: FPS @ BS=1: 172.33 & @ BS=256: 3010.25 versus FPS @ BS=1: 93.84 & @ BS=256: 1785.92 for MobileViT_S
    model = EdgeNeXt(num_landmarks=num_landmarks, depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     **{"classifier_dropout":0})

    return model

'''
Modified from edgenext_small
- XCA for XSA
'''
def edgenext_l_custom(num_landmarks=98):
    model = EdgeNeXt(num_landmarks=num_landmarks, depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     use_spatial_opt=[True,True,True,True],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     **{"classifier_dropout":0})
    return model

'''
Modified from edgenext_small
- XCA for XSA
- Depths [3,4,5,6] instead of [3,3,9,3]
- d2_scales [2,3,4,5] instead of [2,2,3,4]
'''
def edgenext_l_custom_depths(num_landmarks=98):
    model = EdgeNeXt(num_landmarks=num_landmarks, depths=[3, 4, 5, 6], dims=[48, 96, 160, 304], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     use_spatial_opt=[True,True,True,True],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model

'''
Modified from edgenext_small
- Depths [3,4,5,6] instead of [3,3,9,3]
- d2_scales [2,3,4,5] instead of [2,2,3,4]
'''
def edgenext_l_custom_depths_2(num_landmarks):
    model = EdgeNeXt(num_landmarks=num_landmarks, depths=[3, 4, 5, 6], dims=[48, 96, 160, 304], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     use_spatial_opt=[False,False,False,False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model


'''
Modified from edgenext_small
- XCA for XSA
- Depths [4,3,2,1] instead of [3,3,9,3]
- d2_scales [2,3,4,5] instead of [2,2,3,4]
'''
def edgenext_l_custom_depths_3(num_landmarks=98, spatial_attention=False):
    model = EdgeNeXt(num_landmarks=num_landmarks, depths=[2, 2, 2, 2], dims=[24, 96, 160, 304], expan_ratio=4,
                     global_block=[1, 1, 1, 1],
                     global_block_type=['SDTA', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     use_spatial_opt=[spatial_attention, spatial_attention, spatial_attention, spatial_attention],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model

'''
Back to XCA (70ms, 31ms)
- 2,3,2,1 STDA blocks
- 1 Convolution / Stage
'''
def edgenext_l_custom_depths_4(num_landmarks=98, spatial_attention=False):
    model = EdgeNeXt(num_landmarks=num_landmarks, depths=[3, 4, 3, 2], dims=[24, 96, 160, 304], expan_ratio=4,
                     global_block=[2, 3, 2, 1],
                     global_block_type=['SDTA', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     use_spatial_opt=[spatial_attention, spatial_attention, spatial_attention, spatial_attention],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model

'''
More STDA blocks!! (135ms, 41ms)
5, 4, 2, 1 STDA blocks
2 Convs / Last 2 stages (rest 1)
'''
def edgenext_l_custom_depths_5(num_landmarks=98, spatial_attention=False):
    model = EdgeNeXt(num_landmarks=num_landmarks, depths=[6, 5, 4, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                     global_block=[5, 4, 2, 1],
                     global_block_type=['SDTA', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, False, False, False],
                     use_spatial_opt=[spatial_attention, spatial_attention, spatial_attention, spatial_attention],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model

'''
Same as 5 but no convolutions (70ms 21ms)
and reduced first stage channel dim to half (26 vs 48) 
'''
def edgenext_l_custom_depths_6(num_landmarks=98, spatial_attention=False):
    model = EdgeNeXt(num_landmarks=num_landmarks, depths=[5, 4, 2, 1], dims=[24, 96, 160, 304], expan_ratio=4,
                     global_block=[5, 4, 2, 1],
                     global_block_type=['SDTA', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, False, False, False],
                     use_spatial_opt=[spatial_attention, spatial_attention, spatial_attention, spatial_attention],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 3, 4, 5],
                     **{"classifier_dropout":0})
    return model


'''
in_chans: Num of channels of input img
num_classes: Num of prediction head outputs
depths: Num of blocks on each of the stages
global_block: Ammount of Global blocks to use in each stage (STDA)
dims: Channels of the image on each stage
global_block_type: The type of global block in each stage
drop_path_rate: 
heads: number of heads for MHSA (in STDA) in each stage
'''
class EdgeNeXt(nn.Module):
    def __init__(self, in_chans=3, num_landmarks=98,
                 depths=[3, 3, 9, 3], dims=[24, 48, 88, 168],
                 global_block=[0, 0, 0, 3], global_block_type=['None', 'None', 'None', 'SDTA'],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., expan_ratio=4,
                 kernel_sizes=[7, 7, 7, 7], heads=[8, 8, 8, 8], 
                 use_pos_embd_xca=[False, False, False, False], use_pos_embd_global=False,
                 use_spatial_opt=[False, False, False, False],
                 d2_scales=[2, 3, 4, 5], **kwargs):
        super().__init__()
        
        if use_pos_embd_global:
            self.pos_embd = PositionalEncodingFourier(dim=dims[0])
        else:
            self.pos_embd = None
        
        # Downsampling conv layers
        self.downsample_layers = nn.ModuleList()  
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), # 4x4
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        downsample_layer1 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2), # 2x2
        )
        self.downsample_layers.append(downsample_layer1)
        downsample_layer2 = nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
        )
        self.downsample_layers.append(downsample_layer2)
        downsample_layer3 = nn.Sequential(
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2),
        )
        self.downsample_layers.append(downsample_layer3)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()  
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for stage, n_blocks_in_stage in enumerate(depths):
            stage_blocks = []
            for current_block in range(n_blocks_in_stage):

                # depths = [5,4,3,2], global_block = [4,3,2,1]
                # 0 > 5 - 4 - 1 (Conv)
                # 1 > 5 - 4 - 1 ... 
                # final stage 1 : 1 Conv + 4 STDA 
                if current_block > n_blocks_in_stage - global_block[stage] - 1:
                    stage_blocks.append(SDTAEncoder(dim=dims[stage], drop_path=dp_rates[cur + current_block],
                                                    expan_ratio=expan_ratio, scales=d2_scales[stage],
                                                    use_pos_emb=use_pos_embd_xca[stage], num_heads=heads[stage], 
                                                    spatial_opt=use_spatial_opt[stage]))                    
                else:
                    stage_blocks.append(ConvEncoder(dim=dims[stage], drop_path=dp_rates[cur + current_block],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio, kernel_size=kernel_sizes[stage]))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += n_blocks_in_stage
        # for i in range(4):
        #     stage_blocks = []
        #     for n_block in range(depths[i]):
        #         # [3,3,9,3] - [0,1,1,1] - 1 = [0, 1, 7, 1] = 9
        #         # [3,4,5,6] - [0,1,1,1] -1 = [0, 2, 3, 4] = 9
        #         # [4,3,2,1] - [-1,-1,-1,-1] - 1 = [4,3,2,1] = 10 
                
        #         if n_block > depths[i] - global_block[i] - 1:
        #             stage_blocks.append(SDTAEncoder(dim=dims[i], drop_path=dp_rates[cur + n_block],
        #                                             expan_ratio=expan_ratio, scales=d2_scales[i],
        #                                             use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i], spatial_opt=use_spatial_opt[i]))                    
        #         else:
        #             stage_blocks.append(ConvEncoder(dim=dims[i], drop_path=dp_rates[cur + j],
        #                                             layer_scale_init_value=layer_scale_init_value,
        #                                             expan_ratio=expan_ratio, kernel_size=kernel_sizes[i]))

        #     self.stages.append(nn.Sequential(*stage_blocks))
        #     cur += depths[i]


        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final norm layer
        # self.pre_head = nn.Linear(dims[-1], 256)
        
        '''
        HEAD!!
        '''
        # self.head = nn.Linear(dims[-1], num_landmarks*2)
        inhead = dims[-1] 
        self.head = LandmarksLayer(in_features=inhead, num_landmarks=num_landmarks)

        self.apply(self._init_weights)
        self.head_dropout = nn.Dropout(kwargs["classifier_dropout"])
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):  # TODO: MobileViT is using 'kaiming_normal' for initializing conv layers
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
            x = self.stages[i](x) # De aquí puedo sacar las features para detección

        # print(f"Antes del norm: {x.shape}") 
        # # 8, 584, 4, 4
        # TODO - suspect of this last norm (could be doing the -1,1)
        return self.norm(x.mean([-2, -1]))  # Global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.head_dropout(x))
        # x = self.pre_head(x)
        # x = self.head(x)
        return x
