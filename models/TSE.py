import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import _cfg, PatchEmbed
import torch


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class Attent(nn.Module):
    def __init__(self,dim,num_heads=12,qkv_bias=True,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def get_attention_map(self):
        return self.attention_map

    def save_attention_map(self,attention_map):
        self.attention_map = attention_map

    def save_attn_gradients(self,attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self,x,register_hook = False):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)#(1,12,197,197)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,drop=0.):
        super().__init__()
        out_features = in_features
        hidden_features = in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Encoder_layer(nn.Module):
    def __init__(self,dim=768,num_heads=12,mlp_ratio=4.0,qkv_bias = True,qk_scale=None,
                 drop_ratio=0.,attn_drop_ratio=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attent(dim,num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop_ratio,proj_drop=drop_ratio)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim,drop=drop_ratio)

    def forward(self,x,register_hook=False):
        x = x + self.attn(self.norm1(x),register_hook=register_hook)
        x = x+self.mlp(self.norm2(x))
        return x



class TSE(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_chans=3,embed_dim=768,depth=5,num_heads=12,act_layer=None,
                 mlp_ratio=4.,qkv_bias=True,qk_scale=None,drop_ratio=0.,attn_drop_ratio=0.,embed_layer=PatchEmbed,norm_layer=None,
                 device = None):
        super().__init__()
        self.dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_ratio = drop_ratio
        self.attn_drop_rate = attn_drop_ratio
        self.act_layer = act_layer
        self.depth = depth

        self.num_features = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm,eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed_video = embed_layer(img_size=img_size,patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim)
        self.patch_embed_audio = embed_layer(img_size=img_size,patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim)

        num_patches = self.patch_embed_video.num_patches

        self.cls_token_video = nn.Parameter(torch.zeros(1,1,768))
        self.cls_token_audio = nn.Parameter(torch.zeros(1,1,768))
        # position
        self.pos_embed_video = nn.Parameter(torch.zeros(1,num_patches+self.num_tokens,embed_dim))
        self.pos_embed_audio = nn.Parameter(torch.zeros(1,num_patches+self.num_tokens,embed_dim))
        self.pos_drop_video = nn.Dropout(p=drop_ratio)
        self.pos_drop_audio = nn.Dropout(p=drop_ratio)
        # time
        self.time_embed_video = nn.Parameter(torch.zeros(1,embed_dim))
        self.time_embed_audio = nn.Parameter(torch.zeros(1,embed_dim))
        self.time_drop_video = nn.Dropout(p=drop_ratio)
        self.time_drop_audio = nn.Dropout(p=drop_ratio)

        self.video_encoder = nn.Sequential(*[
            Encoder_layer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)
            for _ in range(6)
        ])
        self.device = device
        self.audio_encoder = nn.Sequential(*[
            Encoder_layer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)
            for _ in range(6)
        ])
        self.norm = norm_layer(embed_dim)
        self.audio_proj = nn.Linear(2048, 256)
        #weight init
        nn.init.trunc_normal_(self.pos_embed_video,std=0.02)
        nn.init.trunc_normal_(self.pos_embed_audio,std=0.02)
        nn.init.trunc_normal_(self.cls_token_video,std=0.02)
        nn.init.trunc_normal_(self.cls_token_audio,std=0.02)
        self.apply(_init_vit_weights)


    def forward(self,video,audio):

        audio = self.audio_proj(audio)

        x = self.patch_embed_video(video)
        y = self.patch_embed_video(audio)

        cls_token_video = self.cls_token_video.expand(x.shape[0],-1,-1)
        # cls_token_audio = self.cls_token_audio.expand(y.shape[0],-1,-1)

        x = torch.cat((cls_token_video,x),dim=1)
        y = torch.cat((cls_token_video,y),dim=1)

        # position embed
        x = self.pos_drop_video(x+self.pos_embed_video)
        y = self.pos_drop_video(y+self.pos_embed_audio)

        # time embed
        x = self.time_drop_video(x+self.time_embed_video)
        y = self.time_drop_video(y+self.time_embed_video)

        x = self.video_encoder(x)
        y = self.audio_encoder(y)

        return x,y