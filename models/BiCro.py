import torch.nn as nn
import torch.nn.functional as F
import torch


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads  # 12
        head_dim = dim // num_heads  # 64
        self.scale = qk_scale or head_dim ** -0.5  # 0.125
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # dim = 768
        self.attn_drop = nn.Dropout(attn_drop) # 0
        self.proj = nn.Linear(dim, dim)#（768，768）
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def get_attention_map(self):
        return self.attention_map

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x, register_hook=False):
        B, N, C = x.shape  # (1,197,768)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                        4)  # (3,1,12,197,64)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (1,12,197,64)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (1,12,197,257)
        attn = attn.softmax(dim=-1)#(1,12,197,197)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (1,197,768)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CroAtt(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CroAtt, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)



    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)
        return output


class FeedWard(nn.Module):
    def __init__(self,in_features = None,nets=None,drop_ratio=None):
        super().__init__()
        self.in_features = in_features
        self.nets = nets
        self.drop = nn.Dropout(p=drop_ratio)
        self.w1 = nn.Linear(in_features,nets,bias=True)
        self.w2 = nn.Linear(nets,in_features,bias=True)
        self.act_layer = nn.GELU()

    def forward(self,x):
        x = self.w1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.w2(x)
        x = self.drop(x)

        return x


class BiCroAtt(nn.Module):
    def __init__(self, in_dim1=None, in_dim2=None, k_dim=None, v_dim=None, num_heads=12,
                 qkv_bias = False,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.a2v = CroAtt(in_dim1, in_dim2, k_dim, v_dim, num_heads)
        self.v2a = CroAtt(in_dim2, in_dim1, k_dim, v_dim, num_heads)
        embed_dim = in_dim1
        self.LN = nn.LayerNorm(embed_dim,eps=1e-6)
        self.sat = Attention(dim=embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale = qk_scale,attn_drop=attn_drop,proj_drop=proj_drop)
        self.FF = FeedWard(in_features=embed_dim,nets=1024,out_features=embed_dim,drop_ratio=proj_drop)

    def forward(self,embed_1,embed_2):
        z_embed_1 = self.a2v(embed_1,embed_2)
        z_embed_2 = self.v2a(embed_2,embed_1)

        embed_1 = self.LN(embed_1 + z_embed_1)
        embed_2 = self.LN(embed_2 + embed_2)

        embed_1 = self.LN(embed_1 + self.sat(embed_1))
        embed_2 = self.LN(embed_2 + self.sat(embed_2))

        embed_1 = self.LN(embed_1 + self.FF(embed_1))
        embed_2 = self.LN(embed_2 + self.FF(embed_2))

        return embed_1,embed_2




# text_embeds = torch.rand(2, 32, 768)
# image_embeds = torch.rand(2, 257, 768)
# video_embeds = torch.rand(2, 257, 768)
# audio_embeds = torch.rand(2, 257, 768)
#
# bic = BiCroAtt(
#     in_dim1=768, in_dim2=768, k_dim=64, v_dim=64, num_heads=12,
#     qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.
# )
# va,av = bic(video_embeds,audio_embeds)
# print(va.shape)
# print(av.shape)
#
# t_va,va_t = bic(text_embeds,va)
# print(t_va.shape)
# print(va_t.shape)

# text_embeds = torch.rand(2, 3, 4)
# image_embeds = torch.rand(2,3, 4)
# res = text_embeds+image_embeds
# print(text_embeds)
# print(text_embeds[:,0,:])
