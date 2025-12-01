# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import Mlp
from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
import torch.nn.functional as F
import math
from osgeo import gdal

def trans_tif(image, output_path):
    
    if len(image.shape) == 3:
        bands = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]
    else:
        bands = 1
        height = image.shape[0]
        width = image.shape[1]
    if 'uint8' in image.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'uint16' in image.dtype.name:
        datatype = gdal.GDT_UInt16
    elif 'float64' in image.dtype.name:
        datatype = gdal.GDT_Float64
    else:
        datatype = gdal.GDT_Float32
    # 创建文件
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, width, height, bands, datatype)

    if len(image.shape) == 3:
        for i in range(image.shape[0]):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(image[i])  # 写入第 i+1 个波段
            band.SetNoDataValue(0)  # 设置 NoData 值 (可选
    else:
        band = dataset.GetRasterBand(1).WriteArray(image)


class SimplifiedFrequencyTokens(nn.Module):
    """
    简化版的频率特定tokens生成器，通过结构化初始化确保频率对应性
    """
    def __init__(self, dim, expansion_factor=4, num_bands=3):
        super().__init__()
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.num_bands = num_bands
        
        # 创建具有频率先验的tokens
        self.freq_tokens = nn.ParameterList([
            nn.Parameter(self._initialize_freq_token(i, num_bands, dim * expansion_factor))
            for i in range(num_bands)
        ])
        
        # 简化的频带特定处理层
        self.freq_processors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim * expansion_factor),
                Mlp(
                    in_features=dim * expansion_factor,
                    hidden_features=dim * expansion_factor * 2,
                    out_features=dim * expansion_factor
                )
            ) for _ in range(num_bands)
        ])
        
        # 简化的投影层
        self.projectors = nn.ModuleList([
            nn.Linear(dim * expansion_factor, dim)
            for _ in range(num_bands)
        ])
        
        # 融合层（生成单个CLS token）
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * num_bands),
            nn.Linear(dim * num_bands, dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )
        
        # 初始化
        self.apply(self._init_weights)
    
    def _initialize_freq_token(self, band_id, num_bands, dim):
        """通过结构化模式初始化频率特定的token"""
        token = torch.zeros(1, 1, dim)
        
        # 将维度划分为三个区域，对应低、中、高频
        segment_size = dim // num_bands
        
        if band_id == 0:  # 低频token：平滑变化
            # 使用余弦波形初始化，低频区域有较大值
            indices = torch.arange(0, segment_size)
            low_freq_pattern = torch.cos(indices * (math.pi / segment_size)) * 0.1
            token[0, 0, :segment_size] = low_freq_pattern
            
        elif band_id == 1:  # 中频token：周期性波动
            # 使用中频余弦波形初始化中频区域
            indices = torch.arange(0, segment_size)
            mid_freq_pattern = torch.cos(indices * (3 * math.pi / segment_size)) * 0.1
            token[0, 0, segment_size:2*segment_size] = mid_freq_pattern
            
        else:  # 高频token：快速变化
            # 使用高频余弦波形初始化高频区域
            indices = torch.arange(0, dim-2*segment_size)
            high_freq_pattern = torch.cos(indices * (6 * math.pi / (dim-2*segment_size))) * 0.1
            token[0, 0, 2*segment_size:] = high_freq_pattern
            
        # 添加少量随机噪声以打破对称性
        token += torch.randn_like(token) * 0.01
        return token
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x=None):
        B = 1 if x is None else x.shape[0]
        
        # 处理每个频带的token
        processed_tokens = []
        split_tokens = []
        
        for i in range(self.num_bands):
            # 获取并扩展token
            token = self.freq_tokens[i].expand(B, -1, -1)
            
            # 应用处理层
            processed = self.freq_processors[i](token)
            processed_tokens.append(processed)
            
            # 投影到模型维度
            projected = self.projectors[i](processed)
            split_tokens.append(projected)
        
        # 拼接分离的tokens
        split_tokens = torch.cat(split_tokens, dim=1)  # [B, num_bands, D]
        
        # 融合为单个CLS token
        concat_projections = torch.cat([
            self.projectors[i](processed_tokens[i]) 
            for i in range(self.num_bands)
        ], dim=2)  # [B, 1, num_bands*D]
        
        merged_cls = self.fusion(concat_projections)  # [B, 1, D]
        
        return merged_cls, split_tokens

# 频率滤波器实现
class LowPassFilter(nn.Module):
    """低通滤波器，强调低频信息"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.filter = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim//8),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim//8)
        )
    
    def forward(self, x):
        """
        应用低通滤波
        x: [B, L, D]
        """
        B, L, D = x.shape
        x_trans = x.transpose(1, 2)  # [B, D, L]
        x_filtered = self.filter(x_trans)
        return x_filtered.transpose(1, 2)  # [B, L, D]


class HighPassFilter(nn.Module):
    """高通滤波器，强调高频信息"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 高通滤波通过从原始信号减去低频信号实现
        self.low_filter = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim//8),
            nn.AvgPool1d(kernel_size=5, stride=1, padding=2),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim//8)
        )
        
        self.scale = nn.Parameter(torch.ones(1) * 0.8)
    
    def forward(self, x):
        """应用高通滤波"""
        B, L, D = x.shape
        x_trans = x.transpose(1, 2)  # [B, D, L]
        x_low = self.low_filter(x_trans)
        # 高频 = 原始 - 低频
        x_high = x_trans - self.scale * x_low
        return x_high.transpose(1, 2)  # [B, L, D]


class BandPassFilter(nn.Module):
    """带通滤波器，强调中频信息"""
    def __init__(self, dim, band_id, num_bands):
        super().__init__()
        self.dim = dim
        self.low_thresh = 1.0 / num_bands * band_id
        self.high_thresh = 1.0 / num_bands * (band_id + 1)
        
        # 带通滤波实现
        self.low_filter = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim//8),
            nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        )
        
        self.high_filter = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim//8),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        )
        
        self.scale_low = nn.Parameter(torch.ones(1) * self.low_thresh)
        self.scale_high = nn.Parameter(torch.ones(1) * (1.0 - self.high_thresh))
    
    def forward(self, x):
        """应用带通滤波"""
        B, L, D = x.shape
        x_trans = x.transpose(1, 2)  # [B, D, L]
        
        # 低通滤波
        x_low = self.low_filter(x_trans)
        
        # 高通滤波
        x_high = x_trans - self.high_filter(x_trans)
        
        # 带通 = 原始 - (低频部分 + 高频部分)
        x_mid = x_trans - (self.scale_low * x_low + self.scale_high * x_high)
        
        return x_mid.transpose(1, 2)  # [B, L, D]


class GroupedFrequencyAttention(nn.Module):
    def __init__(self, dim, num_patches, num_groups=3):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches  # 单个组的patch数量
        self.num_groups = num_groups
        

        self.magnitude_attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        

        self.phase_attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Tanh()
        )
        

        self.band_attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 3),  # 3个频带
            nn.Softmax(dim=-1)
        )

        self.cross_gate = nn.Sequential(
            nn.Linear(dim*2, dim//2),
            nn.LayerNorm(dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(dim//2, 1),
            nn.Sigmoid()
        )
        
        # 跨组卷积交互
        self.cross_conv = nn.Conv2d(
            in_channels=dim,          # 输入通道数是D
            out_channels=dim,         # 输出通道数保持D
            kernel_size=(3, 1),       # 3x1卷积核，只在G维度上进行交互
            padding=(1, 0),           # 相应的padding
            groups=1                  # 允许通道间交互
        )
    
    def get_frequency_bands(self, length):
        """创建频带掩码"""
        freq_idx = torch.fft.fftfreq(length) * length
        freq_idx = torch.abs(freq_idx)
        
        # 定义频带范围
        r1, r2 = length // 8, length // 4
        
        # 创建三个频带掩码
        low_mask = (freq_idx <= r1).float()
        mid_mask = ((freq_idx > r1) & (freq_idx <= r2)).float()
        high_mask = (freq_idx > r2).float()
        
        return low_mask, mid_mask, high_mask
    
    def forward(self, x):
        cls_token, seq_tokens = x[:, :1, :], x[:, 1:, :]
        B, GL, D = seq_tokens.shape
        L = self.num_patches  # 单个组的序列长度
        
        x_grouped = seq_tokens.view(B, self.num_groups, L, D)
        global_feat = x_grouped.mean(dim=1, keepdim=True)  # [B,1,L,D]
        
        gate_input = torch.cat([
            x_grouped, 
            global_feat.expand(-1, self.num_groups, -1, -1)
        ], dim=-1)  # [B,G,L,2D]
        
        gate = self.cross_gate(gate_input)  # [B,G,L,1]
        x_grouped = x_grouped * gate + global_feat * (1 - gate)
        
        x_grouped = self.cross_conv(x_grouped.permute(0,3,1,2)).permute(0,2,3,1)  # [B,G,L,D]
        
        outputs = []
        for g in range(self.num_groups):
            group_feat = x_grouped[:, g]  # B, L, D
            
            fft_feat = torch.fft.fft(group_feat.float(), dim=1)  # B, L, D
            magnitude = torch.abs(fft_feat)
            phase = torch.angle(fft_feat)
            
            low_mask, mid_mask, high_mask = self.get_frequency_bands(L)
            masks = [low_mask, mid_mask, high_mask]
            masks = [m.to(magnitude.device) for m in masks]
            
            band_weights = self.band_attention(group_feat.mean(1))  # B, 3
            magnitude_att = self.magnitude_attention(magnitude)  # B, L, D
            phase_att = self.phase_attention(phase)  # B, L, D
            
            magnitude_out = torch.zeros_like(magnitude)
            phase_out = torch.zeros_like(phase)
            
            # 对每个频带分别应用注意力
            for i, mask in enumerate(masks):
                mask = mask.view(1, -1, 1)  # 1, L, 1
                band_weight = band_weights[:, i:i+1].view(-1, 1, 1)  # B, 1, 1
                
                magnitude_band = magnitude * mask * band_weight
                phase_band = phase * mask * band_weight
                
                magnitude_out += magnitude_band
                phase_out += phase_band
            
            magnitude_out = magnitude_out * magnitude_att
            phase_out = phase_out + phase_att

            real = magnitude_out * torch.cos(phase_out)
            imag = magnitude_out * torch.sin(phase_out)
            complex_feat = torch.complex(real, imag)
            
            # 反FFT
            output = torch.fft.ifft(complex_feat, dim=1).real
            outputs.append(output)
        
        attended_seq = torch.cat(outputs, dim=1)  # B, G*L, D
        output = torch.cat([cls_token, attended_seq], dim=1)
        
        return output

class FrequencyAttentionBlock(Block):
    def __init__(
            self,
            dim,
            num_heads,
            num_patches,
            num_groups=3,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        
        # 添加频域注意力相关组件
        self.norm_freq = norm_layer(dim)
        self.freq_attn = GroupedFrequencyAttention(dim, num_patches, num_groups)
        self.fusion_norm = norm_layer(dim)  # 因为要concat两个dim维度的特征
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            act_layer(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        x_att = self.attn(self.norm1(x))
        x_freq = self.freq_attn(self.norm_freq(x))
        combined = torch.cat([x_att, x_freq], dim=-1)
        fused = self.fusion_norm(self.fusion_layer(combined))
        x = x + self.drop_path1(self.ls1(fused))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class MaskedAutoencoderGroupChannelViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, spatial_mask=False,
                 channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
                 channel_embed=256, embed_dim=1024, depth=24, num_heads=16,
                 decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.in_c = in_chans
        self.patch_size = patch_size
        self.channel_groups = channel_groups
        self.spatial_mask = spatial_mask  # Whether to mask all channels of same spatial location
        num_groups = len(channel_groups)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = nn.ModuleList([PatchEmbed(img_size, patch_size, len(group), embed_dim)
                                          for group in channel_groups])
        # self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed[0].num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + 3, embed_dim - channel_embed),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim - decoder_channel_embed),
            requires_grad=False)  # fixed sin-cos embedding
        # Extra channel for decoder to represent special place for cls token
        self.decoder_channel_embed = nn.Parameter(torch.zeros(1, num_groups + 1, decoder_channel_embed),
                                                  requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            FrequencyAttentionBlock(dim=decoder_embed_dim, num_heads=decoder_num_heads, num_patches=num_patches,
                                    num_groups=num_groups, mlp_ratio=mlp_ratio, qkv_bias=True, qk_norm=False,
                                    norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.ModuleList([nn.Linear(decoder_embed_dim, len(group) * patch_size**2)
                                           for group in channel_groups])
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # 替换原始cls_token为频率约束的Jumbo CLS tokens
        self.freq_constrained_cls_encoder = SimplifiedFrequencyTokens(
            dim=embed_dim,  # 获取正确的编码器维度
            expansion_factor=4,
            num_bands=3  # 低、中、高频
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed[0].num_patches ** .5),
                                            cls_token=True, freq_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1],
                                                          torch.arange(len(self.channel_groups)).numpy())
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed[0].num_patches ** .5), cls_token=True, freq_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        dec_channel_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_channel_embed.shape[-1],
                                                              torch.arange(len(self.channel_groups) + 1).numpy())
        self.decoder_channel_embed.data.copy_(torch.from_numpy(dec_channel_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        for patch_embed in self.patch_embed:
            w = patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, C*patch_size**2)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x


    def unpatchify(self, x, p, c):
        """
        x: (N, L, C*patch_size**2)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
        x = torch.einsum('nhwcpq->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
         x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # x is (N, C, H, W)
        b, c, h, w = x.shape

        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D) , N = H * W

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D) G:group->3
        _, G, L, D = x.shape

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, G, 1, cD)   c+p=1
        pos_embed = self.pos_embed[:, 1+3:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, G, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, G, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D)

        if self.spatial_mask:
            # Mask spatial location across all channels (i.e. spatial location as either all/no channels)
            x = x.permute(0, 2, 1, 3).reshape(b, L, -1)  # (N, L, G*D)
            x, mask, ids_restore = self.random_masking(x, mask_ratio)  # (N, 0.25*L, G*D)
            x = x.view(b, x.shape[1], G, D).permute(0, 2, 1, 3).reshape(b, -1, D)  # (N, 0.25*G*L, D)
            mask = mask.repeat(1, G)  # (N, G*L)
            mask = mask.view(b, G, L)
        else:
            # Independently mask each channel (i.e. spatial location has subset of channels visible)
            x, mask, ids_restore = self.random_masking(x.view(b, -1, D), mask_ratio)  # (N, 0.25*G*L, D)
            mask = mask.view(b, G, L)

        # append cls token
        # 传入patch tokens以计算频率约束
        merged_cls, split_tokens = self.freq_constrained_cls_encoder(x)
        
        # 拼接CLS token和split tokens
        x = torch.cat([merged_cls, split_tokens, x], dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, split_tokens.shape[1]

    def forward_decoder(self, x, ids_restore, num_freq_tokens):
        """
        改进的解码器实现，使频率token更有针对性地应用于不同阶段
        
        参数:
            x: 编码器输出的特征，包含CLS token、频率tokens和序列tokens
            ids_restore: 用于恢复原始序列顺序的索引
            num_freq_tokens: 频率token的数量
        """
        # 嵌入tokens
        x = self.decoder_embed(x)
        
        # 分离不同类型的token
        cls_token = x[:, 0:1, :]  # CLS token
        freq_tokens = x[:, 1:1+num_freq_tokens, :]  # 频率tokens
        seq_tokens = x[:, 1+num_freq_tokens:, :]  # 序列tokens
        
        # 获取批次大小和维度信息
        B, _, D = x.shape
        G = len(self.channel_groups)  # 通道组数量
        
        # 处理掩码恢复
        if self.spatial_mask:
            # 处理空间掩码情况
            N, L = ids_restore.shape
            x_ = seq_tokens.view(N, G, -1, seq_tokens.shape[2]).permute(0, 2, 1, 3)  # [N, L/G, G, D]
            _, ml, _, _ = x_.shape
            x_ = x_.reshape(N, ml, G * D)  # [N, L/G, G*D]
            
            # 创建掩码slots
            mask_slots = torch.zeros(N, L - ml, G * D, device=x.device)
            x_ = torch.cat((x_, mask_slots), dim=1)  # 添加掩码slots
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))  # 恢复顺序
            
            # 创建二进制掩码标记被掩盖的位置
            binary_mask = torch.zeros(N, L, device=x.device)
            binary_mask[:, ml:] = 1  # 标记掩码位置
            binary_mask = torch.gather(binary_mask, dim=1, index=ids_restore)  # 恢复顺序
            
            # 重塑为模型需要的格式
            x_ = x_.view(N, L, G, D).permute(0, 2, 1, 3).reshape(N, G*L, D)  # [N, G*L, D]
            seq_tokens = x_
            
            # 扩展掩码以匹配形状
            mask_expanded = binary_mask.unsqueeze(1).repeat(1, G, 1).view(N, -1).unsqueeze(-1).expand(-1, -1, D)
        else:
            # 处理独立掩码情况
            N, L = ids_restore.shape
            visible_len = seq_tokens.shape[1]
            masked_len = ids_restore.shape[1] - visible_len
            
            # 创建掩码slots
            mask_slots = torch.zeros(N, masked_len, D, device=x.device)
            seq_tokens_extended = torch.cat([seq_tokens, mask_slots], dim=1)  # 添加掩码slots
            seq_tokens = torch.gather(seq_tokens_extended, dim=1, 
                                    index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # 恢复顺序
            
            # 创建二进制掩码
            binary_mask = torch.zeros(N, ids_restore.shape[1], device=x.device)
            binary_mask[:, visible_len:] = 1  # 标记掩码位置
            binary_mask = torch.gather(binary_mask, dim=1, index=ids_restore)  # 恢复顺序
            
            # 扩展掩码以用于后续处理
            mask_expanded = binary_mask.unsqueeze(-1).expand(-1, -1, D)
        
        # 为通道和位置添加编码
        channel_embed = self.decoder_channel_embed[:, :-1, :].unsqueeze(2)  # [1, G, 1, cD]
        pos_embed = self.decoder_pos_embed[:, 1:, :].unsqueeze(1)  # [1, 1, L, pD]
        
        # 扩展以匹配形状
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # [1, G, L, cD]
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # [1, G, L, pD]
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # [1, G, L, D]
        pos_channel = pos_channel.view(1, -1, pos_channel.shape[-1])  # [1, G*L, D]
        
        # 为CLS token添加特殊编码
        extra = torch.cat((self.decoder_pos_embed[:, :1, :],
                        self.decoder_channel_embed[:, -1:, :]), dim=-1)  # [1, 1, D]
        
        # 组合所有位置编码
        pos_channel = torch.cat((extra, pos_channel), dim=1)  # [1, 1+G*L, D]
        
        # 添加到tokens
        seq_tokens_with_pos = seq_tokens + pos_channel[:, 1:]  # 不包括CLS的位置编码
        cls_token_with_pos = cls_token + pos_channel[:, :1]  # CLS的位置编码
   
        # 创建频率感知的门控机制
        freq_gates = []
        for i in range(num_freq_tokens):
            # 为每个频率token计算与序列位置的关联性
            freq_token = freq_tokens[:, i:i+1, :]
            
            # 计算注意力分数
            relation = torch.bmm(seq_tokens_with_pos, freq_token.transpose(1, 2)) / math.sqrt(D)  # [B, L, 1]
            gate = torch.sigmoid(relation)  # 转换为0-1之间的门控值
            freq_gates.append(gate)
        
        # 实现解码器的多阶段处理
        decoder_depth = len(self.decoder_blocks)
        stage_size = decoder_depth // 3  # 三个阶段：低频、中频、高频
        
        # 存储各阶段的输出，用于后续计算损失
        stage_outputs = {}
        current_x = torch.cat([cls_token_with_pos, seq_tokens_with_pos], dim=1)
        
        # 通过所有decoder blocks
        for i in range(decoder_depth):
            # 确定当前所在的频率阶段
            if i < stage_size:
                stage_name = "low_freq"
                freq_idx = 0  # 使用低频token
            elif i < 2 * stage_size:
                stage_name = "mid_freq"
                freq_idx = 1  # 使用中频token
            else:
                stage_name = "high_freq"
                freq_idx = 2  # 使用高频token
            
            # 分离当前的CLS token和序列tokens
            current_cls, current_seq = current_x[:, :1, :], current_x[:, 1:, :]
            
            # 应用频率门控，仅对掩码区域
            # 根据当前频率阶段选择合适的频率token和门控
            current_freq_token = freq_tokens[:, freq_idx:freq_idx+1, :]
            current_gate = freq_gates[freq_idx]
            
            # 将频率信息注入到掩码区域
            freq_expansion = current_freq_token.expand(-1, current_seq.shape[1], -1)
            
            # 动态调整门控强度，随着解码深度增加
            gate_strength = 1.0 - (i / decoder_depth) * 0.5  # 从1.0逐渐下降到0.5
            adaptive_gate = current_gate * gate_strength
            
            # 对独立掩码情况进行处理
            current_seq = torch.where(
                mask_expanded.bool(),
                adaptive_gate * freq_expansion + (1 - adaptive_gate) * current_seq,
                current_seq
            )
            
            # 重组并应用transformer block
            current_x = torch.cat([current_cls, current_seq], dim=1)
            current_x = self.decoder_blocks[i](current_x)
            
            # 在每个阶段结束时保存输出
            if (i+1) % stage_size == 0 or i == decoder_depth - 1:
                # 应用normalization
                stage_x = self.decoder_norm(current_x)
                stage_x = stage_x[:, 1:, :]  # 移除CLS token
                
                # 处理成图像块格式
                if self.spatial_mask:
                    N, GL, D = stage_x.shape
                    stage_x = stage_x.view(N, G, GL//G, D)
                else:
                    N, L, D = stage_x.shape
                    stage_x = stage_x.view(N, G, L//G, D)
                
                # 应用预测头
                stage_patches = []
                for g, group in enumerate(self.channel_groups):
                    x_c = stage_x[:, g]
                    dec = self.decoder_pred[g](x_c)
                    dec = dec.view(N, x_c.shape[1], -1, int(self.patch_size**2))
                    dec = torch.einsum('nlcp->nclp', dec)
                    stage_patches.append(dec)
                
                # 保存该阶段的输出
                stage_outputs[stage_name] = torch.cat(stage_patches, dim=1)
        
        # 确保所有阶段的输出都被保存
        # 如果某些阶段没有特定的输出，使用最后一个阶段的输出
        if "low_freq" not in stage_outputs:
            stage_outputs["low_freq"] = stage_outputs["high_freq"]
        if "mid_freq" not in stage_outputs:
            stage_outputs["mid_freq"] = stage_outputs["high_freq"]
        
        return stage_outputs
    
    def forward_freq2d_loss(self, imgs, stage_outputs, mask, current_epoch=None, max_epochs=50):
        # 将图像转换为patches
        target_patches = self.patchify(imgs, self.patch_embed[0].patch_size[0], self.in_c)
        
        if self.norm_pix_loss:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1.e-6) ** .5
        
        target_components = self.decompose_frequency_2d(target_patches)

        for freq_name, target in target_components.items():
            N, L, _ = target.shape
            target = target.view(N, L, self.in_c, -1)  # (N, L, C, p^2)
            target = torch.einsum('nlcp->nclp', target)  # (N, C, L, p^2)
            target_components[freq_name] = target

        stage_losses = {}
        
        for stage_name, pred in stage_outputs.items():
            # 获取当前阶段对应的频率目标
            freq_target = target_components[stage_name]
            
            # 计算均方误差损失
            loss = (pred - freq_target) ** 2
            loss = loss.mean(dim=-1)  # [N, C, L]
            
            stage_loss = 0.0
            num_removed = 0.0
            for i, group in enumerate(self.channel_groups):
                group_loss = loss[:, group, :].mean(dim=1)  # (N, L)
                stage_loss += (group_loss * mask[:, i]).sum()
                num_removed += mask[:, i].sum()
            
            # 归一化损失
            if num_removed > 0:
                stage_loss = stage_loss / num_removed
            stage_losses[stage_name] = stage_loss
        
        # 动态调整各频率阶段的权重
        if current_epoch is not None and max_epochs is not None:
            # 计算训练进度 (0到1之间)
            # 使用后75%的训练时间来调整权重
            effective_progress = max(0.0, (current_epoch - 0.25 * max_epochs) / (0.75 * max_epochs))
            effective_progress = min(1.0, effective_progress)
            
            # 从均衡权重开始，逐渐偏向高频
            # 低频权重: 从1/3降至0.2
            low_weight = 1/3 - (1/3 - 0.1) * effective_progress
            
            # 中频权重: 基本保持稳定
            mid_weight = 1/3
            
            # 高频权重: 从1/3升至0.47
            high_weight = 1/3 + (0.57 - 1/3) * effective_progress
        else:
            # 默认权重 - 如果未提供epoch信息，使用均衡权重
            low_weight = 1/3
            mid_weight = 1/3
            high_weight = 1/3
        
        # 保存当前使用的权重供记录/调试
        self.current_freq_weights = {
            "low_freq": low_weight,
            "mid_freq": mid_weight,
            "high_freq": high_weight
        }
        
        # 计算加权总损失
        total_loss = (stage_losses["low_freq"] * low_weight + 
                    stage_losses["mid_freq"] * mid_weight + 
                    stage_losses["high_freq"] * high_weight)
        return total_loss

    def decompose_frequency_2d(self, imgs):
        """
        使用2D FFT和掩模分解图像频率，直接处理整个批次和所有通道
        
        参数:
            imgs: [B, C, H, W] 图像
        """
        imgs = self.unpatchify(imgs, self.patch_embed[0].patch_size[0], self.in_c)
        B, C, H, W = imgs.shape
        
        # 执行2D FFT
        fft_imgs = torch.fft.fft2(imgs.to(torch.float32))
        fft_imgs = torch.fft.fftshift(fft_imgs, dim=(-2, -1))
        
        # 创建频率掩模
        center_y, center_x = H // 2, W // 2
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        D = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to(imgs.device)
        
        # 定义截止频率
        low_cutoff = min(H, W) // 16
        mid_cutoff = min(H, W) // 8
        
        # 扩展D以匹配batch和channel维度
        D = D.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 创建掩模
        low_mask = (D <= low_cutoff).float()
        mid_mask = ((D > low_cutoff) & (D <= mid_cutoff)).float()
        lowmid_mask = (D <= mid_cutoff).float()  # 直接创建低频+中频掩模
        
        # 应用低频掩模
        fft_low = fft_imgs * low_mask
        fft_low = torch.fft.ifftshift(fft_low, dim=(-2, -1))
        low_freq = torch.fft.ifft2(fft_low).real
        
        # 应用低频+中频掩模
        fft_lowmid = fft_imgs * lowmid_mask
        fft_lowmid = torch.fft.ifftshift(fft_lowmid, dim=(-2, -1))
        lowmid_freq = torch.fft.ifft2(fft_lowmid).real
        
        # 保存各频率分量
        components = {
            "low_freq": self.patchify(low_freq, self.patch_embed[0].patch_size[0], self.in_c),  # 低频
            "mid_freq": self.patchify(lowmid_freq, self.patch_embed[0].patch_size[0], self.in_c),  # 中频
            "high_freq": self.patchify(imgs, self.patch_embed[0].patch_size[0], self.in_c)  # 高频
        }
        
        return components

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, c, H, W]
        pred: [N, L, c*p*p]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs, self.patch_embed[0].patch_size[0], self.in_c)  # (N, L, C*P*P)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        N, L, _ = target.shape
        target = target.view(N, L, self.in_c, -1)  # (N, L, C, p^2)
        target = torch.einsum('nlcp->nclp', target)  # (N, C, L, p^2)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, C, L], mean loss per patch

        total_loss, num_removed = 0., 0.
        for i, group in enumerate(self.channel_groups):
            group_loss = loss[:, group, :].mean(dim=1)  # (N, L)
            total_loss += (group_loss * mask[:, i]).sum()
            num_removed += mask[:, i].sum()  # mean loss on removed patches

        return total_loss/num_removed

    def forward(self, imgs, epoch, mask_ratio=0.75):
        latent, mask, ids_restore, num_freq_tokens = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, num_freq_tokens)  # [N, C, L, p*p]
        loss = self.forward_freq2d_loss(imgs, pred, mask, epoch)
        return loss, pred, mask
    
    def out(low_freq_result, img_patches):
        trans_tif(img_patches[0].cpu().numpy(), '/home/zby/code/FoundFrame/results/imgs1.tif')
        trans_tif(img_patches[1].cpu().numpy(), '/home/zby/code/FoundFrame/results/imgs2.tif')
        trans_tif(img_patches[2].cpu().numpy(), '/home/zby/code/FoundFrame/results/imgs3.tif')
        trans_tif(low_freq_result[0].cpu().numpy(), '/home/zby/code/FoundFrame/results/low_freq_result1.tif')
        trans_tif(low_freq_result[1].cpu().numpy(), '/home/zby/code/FoundFrame/results/low_freq_result2.tif')
        trans_tif(low_freq_result[2].cpu().numpy(), '/home/zby/code/FoundFrame/results/low_freq_result3.tif')


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderGroupChannelViT(
        channel_embed=256, embed_dim=768, depth=12, num_heads=12,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=12, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderGroupChannelViT(
        channel_embed=256, embed_dim=1024, depth=24, num_heads=16,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderGroupChannelViT(
        channel_embed=256, embed_dim=1280, depth=32, num_heads=16,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch8(**kwargs):
    model = MaskedAutoencoderGroupChannelViT(
        channel_embed=256, embed_dim=768, depth=12, num_heads=12,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__ == '__main__':
    input = torch.rand(3,10,96,96)
    model = mae_vit_base_patch8(img_size=96, patch_size=8, in_chans=10,)
    loss, y, mask = model(input, 20)
    print(y.shape)
