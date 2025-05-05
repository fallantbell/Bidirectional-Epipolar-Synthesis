"""
credit to: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, reduce, repeat
import torchvision.transforms as T

from src.main import instantiate_from_config

logger = logging.getLogger(__name__)

def normalize(weight):
    min_val = weight.min()
    max_val = weight.max()
    weight = (weight - min_val) / (max_val - min_val)

    return weight

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768
    

class AdaptiveAttention(nn.Module):
    def __init__(self, block_size, time_len = 3, camera_dim = 30, img_dim = 256):
        super().__init__()
        self.block_size = block_size
        self.camera_dim = camera_dim
        self.img_dim = img_dim
        
        self.fc = nn.Sequential(
            nn.Linear(camera_dim, 2 * camera_dim),
            nn.GELU(),  # nice
            nn.Linear(2 * camera_dim, img_dim**2),
        )
        
    def forward(self, p1=None, p2=None, p3=None):
        # hand-craft assign:
        B = p1.shape[0]
        h = torch.zeros(B, 1, self.block_size, self.block_size).cuda()
        # C 0->1
        if p1 is not None:
            h_01 = self.fc(p1).view(B, 1, self.img_dim, self.img_dim)
            h[:, :, 285:541, 0:256] = h_01  #* query為rgb1 要跟 key rgb0 找關係
        # C 0->2
        if p2 is not None:
            h_02 = self.fc(p2).view(B, 1, self.img_dim, self.img_dim)
            h[:, :, 571:827, 0:256] = h_02  #* query為rgb2 要跟 key rgb0 找關係
        # C 1->2
        if p3 is not None:
            h_12 = self.fc(p3).view(B, 1, self.img_dim, self.img_dim)
            h[:, :, 571:827, 286:542] = h_12 #* query為rgb2 要跟 key rgb1 找關係

        return h
    
class cross_Attention(nn.Module):
    def __init__(self, config,epipolar = None):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        if config.two_cond==True and epipolar!=None:
            self.query2 = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.epipolar = epipolar

        if config.two_cond==True and epipolar!=None:
            self.epipolar = "two_cond"
    
    def forward(self, x, src_encode,forward_map = None,backward_map = None, src_encode0=None):
        
        B, T, C = x.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(src_encode).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(src_encode).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.epipolar == "forward":
            f01 = forward_map[0]
            f01 = repeat(f01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
            att[:,:,0:256,0:256] = att[:,:,0:256,0:256]*f01
        elif self.epipolar == "backward":
            b01 = backward_map[0]
            b01 = b01.permute(0,2,1)
            b01 = repeat(b01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
            att[:,:,0:256,0:256] = att[:,:,0:256,0:256]*b01
        elif self.epipolar == "bidirectional":
            f01 = forward_map[0]
            f01 = repeat(f01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
            b01 = backward_map[0]
            b01 = b01.permute(0,2,1)
            b01 = repeat(b01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
            att[:,:,0:256,0:256] = att[:,:,0:256,0:256]*b01*f01
        elif self.epipolar == "two_cond":
            k2 = self.key(src_encode0).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q2 = self.query2(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v2 = self.value(src_encode0).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            att2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            #* 讓前兩張圖片的condition 隨機作 forward 或 backward
            if torch.randint(0, 2, (1,)).item() == 0: 
                f02 = forward_map[0]
                f02 = repeat(f02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                b12 = backward_map[1]
                b12 = b12.permute(0,2,1)
                b12 = repeat(b12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                att[:,:,0:256,0:256] = att[:,:,0:256,0:256]*b12
                att2[:,:,0:256,0:256] = att2[:,:,0:256,0:256]*f02
            else:
                f12 = forward_map[1]
                f12 = repeat(f12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                b02 = backward_map[0]
                b02 = b02.permute(0,2,1)
                b02 = repeat(b02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                att[:,:,0:256,0:256] = att[:,:,0:256,0:256]*f12
                att2[:,:,0:256,0:256] = att2[:,:,0:256,0:256]*b02
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        if self.epipolar=="two_cond":
            att2 = F.softmax(att2, dim=-1)
            att2 = self.attn_drop(att2)
            y2 = att2 @ v2
            y = (y+y2)/2
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class CausalSelfAttention(nn.Module):
    def __init__(self, config, adaptive,epipolar = None,do_blur = False,mask_cam = False):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"n_embd is {config.n_embd} but n_head is {config.n_head}."
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        
        #* mask shape (827,827)
        #* 其中前 286 為1
        '''
            應該會類似這樣
            1 1 1 0 0
            1 1 1 0 0
            1 1 1 0 0
            1 1 1 1 0
            1 1 1 1 1
            代表 query 只能根據前面的token 去預測下一個token
            例如 query 3 只能看到 key 0~3的資訊
            所以827 代表query 可以看到所有前面的資訊,要去預測第828的token
        '''
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        
        self.adaptive = adaptive
        self.epipolar = epipolar
        self.gaussian_transform = T.GaussianBlur(7,1.5)
        self.do_blur = do_blur
        self.mask_cam = mask_cam
        
    def forward(self, x, x_kv, h, layer_past=None,forward_map = None,backward_map = None,return_attn=False):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x_kv).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x_kv).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 不重要,沒用到
        if self.adaptive:
            att = h[:,:,:T,:T] + att
        
        epipolar_attn_map = None
        bi_epi_ratio = None
        att_for = None
        if self.epipolar!=None:
            #* 在做epipolar 之前先把attention 經由softmax 全為正數, 有負數有可能會出錯
            att[:,:,285:541, 0:256] = F.softmax(att[:,:,285:541, 0:256],dim=-1)
            att[:, :, 571:827, 0:256] = F.softmax(att[:,:,571:827, 0:256],dim=-1)
            att[:, :, 571:827, 286:542] = F.softmax(att[:,:,571:827, 286:542],dim=-1)

            # 測試用visualization
            if return_attn:
                epipolar_attn_map = att.clone()
                epipolar_attn_map = epipolar_attn_map.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

            if self.epipolar == "forward":
                f01, f02, f12 = forward_map
                f01 = repeat(f01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                f02 = repeat(f02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                f12 = repeat(f12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                #* 根據forward epipolar map 對 img01,img02,img12 之間的attention 做reweighting
                att[:,:,285:541, 0:256] = att[:,:,285:541, 0:256]*f01[:,:,:min(T-285,256),...] 
                if T>571:
                    att[:, :, 571:827, 0:256] = att[:, :, 571:827, 0:256]*f02[:,:,:T-571,...]
                    att[:, :, 571:827, 286:542] = att[:, :, 571:827, 286:542]*f12[:,:,:T-571,...]

            elif self.epipolar == "backward":
                b01, b02, b12 = backward_map
                b01 = b01.permute(0,2,1)
                b02 = b02.permute(0,2,1)
                b12 = b12.permute(0,2,1)
                b01 = repeat(b01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                b02 = repeat(b02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                b12 = repeat(b12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                #* 根據backward epipolar map 對 img01,img02,img12 之間的attention 做reweighting
                att[:,:,285:541, 0:256] = att[:,:,285:541, 0:256]*b01[:,:,:min(T-285,256),...]
                if T>571:
                    att[:, :, 571:827, 0:256] = att[:, :, 571:827, 0:256]*b02[:,:,:T-571,...]
                    att[:, :, 571:827, 286:542] = att[:, :, 571:827, 286:542]*b12[:,:,:T-571,...]

            elif self.epipolar == "bidirectional":
                f01, f02, f12 = forward_map

                b01, b02, b12 = backward_map
                b01 = b01.permute(0,2,1)
                b02 = b02.permute(0,2,1)
                b12 = b12.permute(0,2,1)

                bi_12 = f12*b12
                bi_12_mask = (bi_12>=0.1).float() #* 用來看epipolar line 所選定的區域, epipolar 內的區域為1 其餘為0

                f01 = repeat(f01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                f02 = repeat(f02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                f12 = repeat(f12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                b01 = repeat(b01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                b02 = repeat(b02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                b12 = repeat(b12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])

                bi_12_mask = repeat(bi_12_mask,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                if T==827:
                    #* token 對整張圖的attention平均值
                    att_mean = att[:,:,571:827,286:542].mean(dim=-1)
                    
                    att_mask = att[:, :, 571:827, 286:542]*bi_12_mask[:,:,:T-571,...]
                    #* bidirection epipolar 所在區域的token 數 (b nh hw)
                    bi_12_mask_sum = bi_12_mask[:,:,:T-571,...].sum(dim=-1)
                    #* 區域內attention 的總和 / 區域token 數 = epipolar 範圍內的attention 平均值
                    att_mask_mean = att_mask.sum(dim=-1) / bi_12_mask_sum

                    #* epipolar 範圍內attention 佔全體的比例，比例越高代表這個token 有更好的找到對應epipolar的區塊，也代表這個token 更可信
                    #* (b hw)
                    bi_epi_ratio = att_mask_mean.mean(dim=1) / att_mean.mean(dim=1)
                elif T==541:
                    att_mean = att[:,:,285:541, 0:256].mean(dim=-1)
                    att_mask = att[:, :,285:541, 0:256]*bi_12_mask[:,:,:min(T-285,256),...]
                    bi_12_mask_sum = bi_12_mask[:,:,:min(T-285,256),...].sum(dim=-1)
                    att_mask_mean = att_mask.sum(dim=-1) / bi_12_mask_sum
                    bi_epi_ratio = att_mask_mean.mean(dim=1) / att_mean.mean(dim=1)
                else:
                    #* 只有第二張image 或 第三張image 全部token 生成完畢才需要計算ratio 回傳
                    #* 所以只需要看 T=541 跟 T=827 就好，其他不需要重複計算
                    bi_epi_ratio = None

                #* att_for, 測試用 visualization
                att_for = att.clone()
                att_for[:,:,285:541, 0:256] = att_for[:,:,285:541, 0:256]*f01[:,:,:min(T-285,256),...]
                att[:,:,285:541, 0:256] = att[:,:,285:541, 0:256]*b01[:,:,:min(T-285,256),...]*f01[:,:,:min(T-285,256),...]
                if T>571:
                    att[:, :, 571:827, 0:256] = att[:, :, 571:827, 0:256]*b02[:,:,:T-571,...]*f02[:,:,:T-571,...]
                    att[:, :, 571:827, 286:542] = att[:, :, 571:827, 286:542]*b12[:,:,:T-571,...]*f12[:,:,:T-571,...]
                    att_for[:, :, 571:827, 0:256] = att_for[:, :, 571:827, 0:256]*f02[:,:,:T-571,...]
                    att_for[:, :, 571:827, 286:542] = att_for[:, :, 571:827, 286:542]*f12[:,:,:T-571,...]
            else:
                raise AssertionError("Invalid type for epipolar")
        
            if self.mask_cam:
                #* 將 image 與 camera 之間的attention mask 掉
                att[:,:,285:541, 256:] = float('-inf')
                att[:,:,571:827, 256:286] = float('-inf')
                att[:,:,571:827, 542:] = float('-inf')
                att_for[:,:,285:541, 256:] = float('-inf')
                att_for[:,:,571:827, 256:286] = float('-inf')
                att_for[:,:,571:827, 542:] = float('-inf')

        att_weight = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        if self.epipolar!=None:
            att_weight[:, :, 571:827, 0:256]= F.softmax(att[:, :, 571:827, 0:256], dim=-1)
            att_weight[:, :, 571:827, 286:542]= F.softmax(att[:, :, 571:827, 286:542], dim=-1)
        att_weight = F.softmax(att_weight, dim=-1)

        #* 測試用visualization -----------------------------------------
        if self.epipolar!=None:
            att_weight_for = att_for.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            if self.epipolar!=None:
                att_weight_for[:, :, 571:827, 0:256]= F.softmax(att_for[:, :, 571:827, 0:256], dim=-1)
                att_weight_for[:, :, 571:827, 286:542]= F.softmax(att_for[:, :, 571:827, 286:542], dim=-1)
            att_weight_for = F.softmax(att_weight_for, dim=-1)
        #* -------------------------------------------------------------

        att = self.attn_drop(att_weight)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        if return_attn:
            return y, epipolar_attn_map,att_weight,att_weight_for,bi_epi_ratio
        else:
            if self.epipolar!=None:
                return y,[],[],[],bi_epi_ratio
            else:
                return y,[],[],[],[]

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config, adaptive,epipolar=None,do_blur=False,mask_cam=False,selfremain=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, adaptive,epipolar,do_blur,mask_cam)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.selfremain = selfremain

    def forward(self, x,x_kv, p,forward_map=None,backward_map=None,return_attn=False):
        out, epipolar_attn_map, attn_weight,attn_weight_for,bi_epi_ratio = self.attn(self.ln1(x),self.ln1(x_kv),p,
                                forward_map = forward_map,
                                backward_map = backward_map,
                                return_attn = return_attn)
        
        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x, epipolar_attn_map, attn_weight,attn_weight_for,bi_epi_ratio


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, time_len, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0,
                 input_vocab_size=None,epipolar=None,do_cross=False,sep_pe = False,
                 two_cond = False,do_blur=False,mask_cam=False,srcimg_pe=True,selfremain=False):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked,two_cond = two_cond)
        # input embedding stem
        in_vocab_size = vocab_size if not input_vocab_size else input_vocab_size
        self.tok_emb = nn.Embedding(in_vocab_size, config.n_embd)
        
        # Locality
        self.locality = AdaptiveAttention(config.block_size)
        
        # init pos embedding
        self.time_len = time_len
        self.frame_emb = nn.Parameter(torch.zeros(1, 256, config.n_embd))
        self.camera_emb = nn.Parameter(torch.zeros(1, 30, config.n_embd))
        self.role_emb = None
        
        self.time_emb = nn.Parameter(data=get_sinusoid_encoding(n_position=block_size, d_hid=config.n_embd), requires_grad=False)
        
        #* 試試讓condition 跟 target 用不同的 position encode
        self.sep_pe = sep_pe
        if sep_pe == True:
            self.frame_emb2 = nn.Parameter(torch.zeros(1, 256, config.n_embd))
            self.camera_emb2 = nn.Parameter(torch.zeros(1, 30, config.n_embd))
            self.time_emb2 = nn.Parameter(data=get_sinusoid_encoding(n_position=block_size, d_hid=config.n_embd), requires_grad=False)

        # dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer 
        self.blocks = nn.ModuleList()

        self.epipolar = epipolar
        if self.epipolar == "":
            self.epipolar = None

        if self.epipolar == None:
            #* 做原本的locality
            for _ in range(int(config.n_layer // 2)):
                self.blocks.append(Block(config, adaptive = True))
                self.blocks.append(Block(config, adaptive = False))
        else:
            for _ in range(int(config.n_layer // 2)):
                self.blocks.append(Block(config, adaptive = False,epipolar=epipolar,do_blur=do_blur,mask_cam=mask_cam,selfremain=selfremain))
                self.blocks.append(Block(config, adaptive = False,epipolar=None))
            
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_epipolar_tensor(self,b,h,w,k,src_w2c,target_w2c):
        '''
            回傳 target image 中每個點對應到src image 的weighted map, shape = (hw,hw) 
        '''

        H = h
        W = H*16/9  #* 原始圖像為 16:9

        k = k.to(dtype=torch.float32)
        src_w2c=src_w2c.to(dtype=torch.float32)
        target_w2c=target_w2c.to(dtype=torch.float32)

        #* unormalize intrinsic 

        k[:,0] = k[:,0]*W
        k[:,1] = k[:,1]*H

        k[:,0,2] = h/2
        k[:,1,2] = h/2

        device = k.device

        #* 創建 h*w 的 uv map
        x_coords, y_coords = torch.meshgrid(torch.arange(h), torch.arange(w))
        coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
        coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
        coords_tensor = coords_tensor.to(dtype=torch.float32)
        coords_tensor = repeat(coords_tensor, 'HW p -> b HW p', b=b)

        x_coords = x_coords.to(device)
        y_coords = y_coords.to(device)
        coords_tensor = coords_tensor.to(device)

        k_3x3 = k[:,0:3,0:3]
        src_w2c_r = src_w2c[:,0:3,0:3]
        src_w2c_t = src_w2c[:,0:3,3]
        target_w2c_r = target_w2c[:,0:3,0:3]
        target_w2c_t = target_w2c[:,0:3,3]
        target_c2w_r = torch.linalg.inv(target_w2c_r)
        target_c2w_t = -target_w2c_t

        cx = k_3x3[:,0,2].view(b, 1)
        cy = k_3x3[:,1,2].view(b, 1)
        fx = k_3x3[:,0,0].view(b, 1)
        fy = k_3x3[:,1,1].view(b, 1)
        coords_tensor[...,0] = (coords_tensor[...,0]-cx)/fx
        coords_tensor[...,1] = (coords_tensor[...,1]-cy)/fy

        #* 做 H*W 個點的運算
        coords_tensor = rearrange(coords_tensor, 'b hw p -> b p hw') 
        point_3d_world = torch.matmul(target_c2w_r,coords_tensor)              #* 相機坐標系 -> 世界座標
        point_3d_world = point_3d_world + target_c2w_t.unsqueeze(-1)           #* 相機坐標系 -> 世界座標
        point_2d = torch.matmul(src_w2c_r,point_3d_world)                #* 世界座標 -> 相機座標
        point_2d = point_2d + src_w2c_t.unsqueeze(-1)                    #* 世界座標 -> 相機座標
        pi_to_j = torch.matmul(k_3x3,point_2d)                              #* 相機座標 -> 平面座標

        #* 原點的計算
        oi = torch.zeros(3).to(dtype=torch.float32)
        oi = repeat(oi, 'p -> b p', b=b)
        oi = oi.unsqueeze(-1)
        oi = oi.to(device)
        point_3d_world = torch.matmul(target_c2w_r,oi)
        point_3d_world = point_3d_world + target_c2w_t.unsqueeze(-1)  
        point_2d = torch.matmul(src_w2c_r,point_3d_world)
        point_2d = point_2d + src_w2c_t.unsqueeze(-1)  
        oi_to_j = torch.matmul(k_3x3,point_2d)
        oi_to_j = rearrange(oi_to_j, 'b c p -> b p c') #* (b,3,1) -> (b,1,3)

        #* 除以深度
        pi_to_j_unnormalize = rearrange(pi_to_j, 'b p hw -> b hw p') 
        
        #* target image 每個點投影到src image 的位置, (b,hw,3)
        pi_to_j = pi_to_j_unnormalize / (pi_to_j_unnormalize[..., -1:] + 1e-6) 
        #* target image 原點投影到src image 的位置, (b,1,3)
        oi_to_j = oi_to_j / oi_to_j[..., -1:]  


        #* 計算feature map 每個點到每個 epipolar line 的距離
        coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
        coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
        coords_tensor = coords_tensor.to(dtype=torch.float32) 
        coords_tensor = repeat(coords_tensor, 'HW p -> b HW p', b=b)
        coords_tensor = coords_tensor.to(device)

        oi_to_pi = pi_to_j - oi_to_j            #* h*w 個 epipolar line (b,hw,3)
        oi_to_coord = coords_tensor - oi_to_j   #* h*w 個點   (b,hw,3)

        ''''
            #* 這裡做擴展
            oi_to_pi    [0,0,0]     ->      oi_to_pi_repeat     [0,0,0]
                        [1,1,1]                                 [0,0,0]
                        [2,2,2]                                 [1,1,1]
                                                                [1,1,1]
                                                                .
                                                                .
                                                                .

            oi_to_coord     [0,0,0]     ->      oi_to_coord_repeat      [0,0,0]
                            [1,1,1]                                     [1,1,1]
                            [2,2,2]                                     [2,2,2]
                                                                        [0,0,0]
                                                                        .
                                                                        .
                                                                        .
        '''
        oi_to_pi_repeat = repeat(oi_to_pi, 'b i j -> b i (repeat j)',repeat = h*w)
        oi_to_pi_repeat = rearrange(oi_to_pi_repeat,"b i (repeat j) -> b (i repeat) j", repeat = h*w)
        oi_to_coord_repeat = repeat(oi_to_coord, 'b i j -> b (repeat i) j',repeat = h*w)

        #* src image 上 hw 個點與 hw 條epipolar line 的距離
        #* 先用外積算面積, 再除以底的長度, 得到距離
        area = torch.cross(oi_to_pi_repeat,oi_to_coord_repeat,dim=-1)     #* (b,hw*hw,3)
        area = torch.norm(area,dim=-1 ,p=2)
        vector_len = torch.norm(oi_to_pi_repeat, dim=-1, p=2)
        distance = area/vector_len

        #* 50 0.5 基礎的epipolar
        #* 5 0.75 flexible epipolar, 希望epipolar 可以多看一點
        #* 距離越小權重越大
        distance_weight = 1 - torch.sigmoid(5*(distance-0.75)) # 50 0.5
        # distance_weight = 1 - torch.sigmoid(steep*(distance-0.05*H)) # 50 0.5

        epipolar_map = rearrange(distance_weight,"b (hw hw2) -> b hw hw2",hw = h*w)

        #* 若是weight map 最大權重<0.5, 代表epipolar line 沒有找到, 權重就全設成1, 不特別加權
        max_values, _ = torch.max(epipolar_map, dim=-1)
        mask = max_values < 0.5
        epipolar_map[mask.unsqueeze(-1).expand_as(epipolar_map)] = 1

        #* 回傳 (b,hw,hw2) 的weighted map
        return epipolar_map
    
    def iter_forward(self, dc_emb, z_emb, p,k=None,w2c=None, embeddings=None, targets=None, return_layers=False):
        
        token_embeddings_dc = dc_emb
        token_embeddings_z = z_emb
        token_embeddings = torch.cat([token_embeddings_dc, token_embeddings_z], 1)
        token_embeddings = token_embeddings[:, :-1, :] # remove the last one

        # drop out for teacher
        token_embeddings = self.drop(token_embeddings)
        
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        #* 對image 與 camera 分別做embedding
        role_emb = []
        for _ in range(self.time_len-1):
            role_emb.append(self.frame_emb)
            role_emb.append(self.camera_emb)

        role_emb.append(self.frame_emb)
        role_emb = torch.cat(role_emb, 1)

        #* role emb shape (1,827,1024)
        #* time emb shape (1,827,1024)
        
        role_embeddings = role_emb[:, :t, :] # each position maps to a (learnable) vector
        time_embeddings = self.time_emb[:, :t, :] # each position maps to a (learnable) vector
        
        #* x shape (B,827,1024)
        x = token_embeddings + role_embeddings + time_embeddings

        origin_x = x.clone()

        if return_layers:
            layers = [x]
            for block in self.blocks:
                x = block(x)
                layers.append(x)
            return layers
        
        #* 計算epipolar map [forward,backward,bidirectional]
        batch = x.shape[0]
        forward_epipolar_map = None
        backward_epipolar_map = None
        if self.epipolar!=None:
            if self.epipolar == 'forward' or self.epipolar == 'bidirectional':
                w2c_0 = w2c[:,0]
                w2c_1 = w2c[:,1]
                w2c_2 = w2c[:,2]
                #* 256x256 的weighted map
                f01 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_0,w2c_1)
                f02 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_0,w2c_2)
                f12 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_1,w2c_2)
                forward_epipolar_map = [f01,f02,f12]
            if self.epipolar == 'backward' or self.epipolar == 'bidirectional':
                w2c_0 = w2c[:,0]
                w2c_1 = w2c[:,1]
                w2c_2 = w2c[:,2]
                b01 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_1,w2c_0)
                b02 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_2,w2c_0)
                b12 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_2,w2c_1)
                backward_epipolar_map = [b01,b02,b12]

        # locality
        #* 不重要, 沒用上
        p1, p2, p3 = p
        h = self.locality(p1, p2, p3)
        # h = h.repeat(x.shape[0], 1, 1, 1)

        if self.epipolar!=None:
            for i in range(len(self.blocks)):
                if i%2==0:
                    #* 偶數層拿 origin_x 做cross attention
                    x,_,_,_ = self.blocks[i](x, origin_x,h,
                                            forward_map = forward_epipolar_map,
                                            backward_map = backward_epipolar_map)
                else:
                    #* 奇數層自己對自己做 self attention
                    x,_,_,_ = self.blocks[i](x, x, h,
                                            forward_map = forward_epipolar_map,
                                            backward_map = backward_epipolar_map)
        else:
            for block in self.blocks:
                x = block(x, h,forward_map = forward_epipolar_map,backward_map = backward_epipolar_map)

        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    
    def test(self, dc_emb, z_indices, p,forward_epipolar_map=None,backward_epipolar_map=None, embeddings=None, 
             targets=None, return_layers=False, return_bias=False,
                return_attn = False
             ):
        token_embeddings_dc = dc_emb

        # add the token embedding with z_indices
        token_embeddings_z = self.tok_emb(z_indices)
        token_embeddings = torch.cat([token_embeddings_dc, token_embeddings_z], 1)
        
        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        role_emb = []
        for _ in range(self.time_len-1):
            role_emb.append(self.frame_emb)
            role_emb.append(self.camera_emb)

        role_emb.append(self.frame_emb)
        role_emb = torch.cat(role_emb, 1)
        
        role_embeddings = role_emb[:, :t, :] # each position maps to a (learnable) vector
        time_embeddings = self.time_emb[:, :t, :] # each position maps to a (learnable) vector
        
        x = token_embeddings + role_embeddings + time_embeddings

        origin_x = x.clone()

        # locality
        p1, p2, p3 = p
        h = self.locality(p1, p2, p3)
        h = h.repeat(x.shape[0], 1, 1, 1) #* (1,1,827,827)
        
        #* x shape (1,286,1024)
        #* f01 shape (1,256,256)
        # for block in self.blocks:
        #     x = block(x, h)

        attn_weights = []
        attn_weights_for = []
        epipolar_attn_maps = []

        bi_epi_ratio = None
        if self.epipolar!=None:
            for i in range(len(self.blocks)):
                if i%2==0:
                    x, epipolar_attn_map,attn_weight,attn_weight_for,ratio = self.blocks[i](x, origin_x,h,
                                            forward_map = forward_epipolar_map,
                                            backward_map = backward_epipolar_map,
                                            return_attn = return_attn
                                            )
                    attn_weights.append(attn_weight)
                    attn_weights_for.append(attn_weight_for)
                    epipolar_attn_maps.append(epipolar_attn_map)
                    bi_epi_ratio = ratio
                else:
                    x,_,_,_,_ = self.blocks[i](x, x, h,
                                            forward_map = forward_epipolar_map,
                                            backward_map = backward_epipolar_map)
        else:
            for block in self.blocks:
                x,_,_,_ = block(x,x, h,forward_map = forward_epipolar_map,backward_map = backward_epipolar_map)
            
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits,epipolar_attn_maps,attn_weights,attn_weights_for,bi_epi_ratio
        if return_bias:
            return logits, h
        else:
            return logits, loss

class DummyGPT(nn.Module):
    # for debugging
    def __init__(self, add_value=1):
        super().__init__()
        self.add_value = add_value

    def forward(self, idx):
        return idx + self.add_value, None

#### sampling utils
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x