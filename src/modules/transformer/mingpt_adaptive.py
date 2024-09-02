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
    def __init__(self, config, adaptive,epipolar = None,do_blur = False):
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
        # print(f"mask shape = {mask.shape}")
        # print(f"unmask = {config.n_unmasked}")
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
        
    def forward(self, x, h, layer_past=None,forward_map = None,backward_map = None):
        B, T, C = x.size()
        # print(f"T = {T}")

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if self.adaptive:
            att = h[:,:,:T,:T] + att
        
        if self.epipolar!=None:
            if self.epipolar == "forward":
                # forward_map = repeat(forward_map,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                f01, f02, f12 = forward_map
                f01 = repeat(f01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                f02 = repeat(f02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                f12 = repeat(f12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                att[:,:,285:541, 0:256] = att[:,:,285:541, 0:256]*f01[:,:,:min(T-285,256),...] #*query 能找到 key一定可以找到全部
                if T>571:
                    att[:, :, 571:827, 0:256] = att[:, :, 571:827, 0:256]*f02[:,:,:T-571,...]
                    att[:, :, 571:827, 286:542] = att[:, :, 571:827, 286:542]*f12[:,:,:T-571,...]
                # att = att*forward_map
            elif self.epipolar == "backward":
                # backward_map = repeat(backward_map,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                b01, b02, b12 = backward_map
                b01 = b01.permute(0,2,1)
                b02 = b02.permute(0,2,1)
                b12 = b12.permute(0,2,1)
                b01 = repeat(b01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                b02 = repeat(b02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                b12 = repeat(b12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                att[:,:,285:541, 0:256] = att[:,:,285:541, 0:256]*b01[:,:,:min(T-285,256),...]
                if T>571:
                    att[:, :, 571:827, 0:256] = att[:, :, 571:827, 0:256]*b02[:,:,:T-571,...]
                    att[:, :, 571:827, 286:542] = att[:, :, 571:827, 286:542]*b12[:,:,:T-571,...]
                # att = att*backward_map
            elif self.epipolar == "bidirectional":
                # forward_map = repeat(forward_map,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                # backward_map = repeat(backward_map,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                f01, f02, f12 = forward_map

                b01, b02, b12 = backward_map
                b01 = b01.permute(0,2,1)
                b02 = b02.permute(0,2,1)
                b12 = b12.permute(0,2,1)

                if self.do_blur:

                    bi01 = f01 * b01
                    bi02 = f02 * b02
                    bi12 = f12 * b12
                    bi01 = rearrange(bi01,'b hw (h w) -> (b hw) 1 h w',h = 16)
                    bi02 = rearrange(bi02,'b hw (h w) -> (b hw) 1 h w',h = 16)
                    bi12 = rearrange(bi12,'b hw (h w) -> (b hw) 1 h w',h = 16)

                    blur_epipolar_map01 = self.gaussian_transform(bi01)
                    blur_epipolar_map02 = self.gaussian_transform(bi02)
                    blur_epipolar_map12 = self.gaussian_transform(bi12)
                    blur_epipolar_map01 = normalize(blur_epipolar_map01)
                    blur_epipolar_map02 = normalize(blur_epipolar_map02)
                    blur_epipolar_map12 = normalize(blur_epipolar_map12)
                    blur_epipolar_map01 = blur_epipolar_map01.clamp(0,1).squeeze(1)
                    blur_epipolar_map02 = blur_epipolar_map02.clamp(0,1).squeeze(1)
                    blur_epipolar_map12 = blur_epipolar_map12.clamp(0,1).squeeze(1)
                    blur_epipolar_map01 = rearrange(blur_epipolar_map01,'(b hw) h w -> b hw (h w)',hw=16*16)
                    blur_epipolar_map02 = rearrange(blur_epipolar_map02,'(b hw) h w -> b hw (h w)',hw=16*16)
                    blur_epipolar_map12 = rearrange(blur_epipolar_map12,'(b hw) h w -> b hw (h w)',hw=16*16)

                    blur01 = blur_epipolar_map01 * f01 * b01
                    blur02 = blur_epipolar_map02 * f02 * b02
                    blur12 = blur_epipolar_map12 * f12 * b12

                    blur01 = repeat(blur01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                    blur02 = repeat(blur02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                    blur12 = repeat(blur12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])

                    att[:,:,285:541, 0:256] = att[:,:,285:541, 0:256]*blur01[:,:,:min(T-285,256),...]
                    if T>571:
                        att[:, :, 571:827, 0:256] = att[:, :, 571:827, 0:256]*blur02[:,:,:T-571,...]
                        att[:, :, 571:827, 286:542] = att[:, :, 571:827, 286:542]*blur12[:,:,:T-571,...]

                else:

                    f01 = repeat(f01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                    f02 = repeat(f02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                    f12 = repeat(f12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                    b01 = repeat(b01,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                    b02 = repeat(b02,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])
                    b12 = repeat(b12,'b hw hw2 -> b nh hw hw2',nh=att.shape[1])


                    att[:,:,285:541, 0:256] = att[:,:,285:541, 0:256]*b01[:,:,:min(T-285,256),...]*f01[:,:,:min(T-285,256),...]
                    if T>571:
                        att[:, :, 571:827, 0:256] = att[:, :, 571:827, 0:256]*b02[:,:,:T-571,...]*f02[:,:,:T-571,...]
                        att[:, :, 571:827, 286:542] = att[:, :, 571:827, 286:542]*b12[:,:,:T-571,...]*f12[:,:,:T-571,...]
                # att = att*forward_map
                # att = att*backward_map
            elif self.epipolar == "token_change":
                pass
            else:
                raise AssertionError("Invalid type for epipolar")
        
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config, adaptive,epipolar=None,do_blur=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, adaptive,epipolar,do_blur)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, p,forward_map=None,backward_map=None):
        x = x + self.attn(self.ln1(x), p,forward_map = forward_map,backward_map = backward_map)
        x = x + self.mlp(self.ln2(x))
        return x
    
class Block_cross(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config,epipolar=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = cross_Attention(config,epipolar)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, src_encode,forward_map=None,backward_map=None,src_encode0=None):
        if src_encode0 != None:
            x = x + self.attn(self.ln1(x), self.ln1(src_encode),
                              forward_map = forward_map,backward_map = backward_map,
                              src_encode0 = self.ln1(src_encode0))
        else:
            x = x + self.attn(self.ln1(x), self.ln1(src_encode),forward_map = forward_map,backward_map = backward_map)
        
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, time_len, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0,
                 input_vocab_size=None,epipolar=None,do_cross=False,sep_pe = False,
                 two_cond = False,do_blur=False):
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

        if do_cross == True:
            self.start_token = nn.Parameter(torch.zeros(1, 286, config.n_embd))
        
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

        self.do_cross = do_cross

        print(f"do blur = {do_blur}")
        
        if do_cross==False:

            if self.epipolar == None:
                #* 做原本的locality
                for _ in range(int(config.n_layer // 2)):
                    self.blocks.append(Block(config, adaptive = True))
                    self.blocks.append(Block(config, adaptive = False))
            else:
                #! 不做locality 改作epipolar
                if epipolar == "alternately":
                    for i in range(int(config.n_layer // 2)):
                        if i%2==0:
                            self.blocks.append(Block(config, adaptive = False,epipolar="forward"))
                        else:
                            self.blocks.append(Block(config, adaptive = False,epipolar="backward"))
                        self.blocks.append(Block(config, adaptive = False,epipolar=None))
                else:
                    for _ in range(int(config.n_layer // 2)):
                        self.blocks.append(Block(config, adaptive = False,epipolar=epipolar,do_blur=do_blur))
                        self.blocks.append(Block(config, adaptive = False,epipolar=None))
        
        elif do_cross == True:
            '''
                recon:          前 forward 後 backward
                just forward:   前 forward 後 forward
                預設cross:      前 bidirectional 後 bidirectional
            '''
            # for _ in range(int(config.n_layer // 2)):
            #     #* cross attn 跟 self attn 交替
            #     self.blocks.append(Block_cross(config,epipolar="bidirectional"))
            #     self.blocks.append(Block_cross(config,epipolar=None))

            for _ in range(int(config.n_layer // 2)-3): # int(config.n_layer // 2)//2
                #* cross attn 跟 self attn 交替
                self.blocks.append(Block_cross(config,epipolar="forward"))
                self.blocks.append(Block_cross(config,epipolar=None))

            self.blocks2 = nn.ModuleList()
            for _ in range(3):
                #* cross attn 跟 self attn 交替
                self.blocks2.append(Block_cross(config,epipolar="forward"))
                self.blocks2.append(Block_cross(config,epipolar=None))

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
        pi_to_j = pi_to_j_unnormalize / (pi_to_j_unnormalize[..., -1:] + 1e-6)   #* (b,hw,3)
        # pi_to_j = pi_to_j_unnormalize / pi_to_j_unnormalize[..., -1:]
        oi_to_j = oi_to_j / oi_to_j[..., -1:]   #* (b,1,3)

        # print(f"pi_to_j: {pi_to_j[0,9]}")
        # print(f"oi_to_j: {oi_to_j[0,0]}")

        #* 計算feature map 每個點到每個 epipolar line 的距離
        coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
        coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
        coords_tensor = coords_tensor.to(dtype=torch.float32) # (4096,3)
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


        area = torch.cross(oi_to_pi_repeat,oi_to_coord_repeat,dim=-1)     #* (b,hw*hw,3)
        area = torch.norm(area,dim=-1 ,p=2)
        vector_len = torch.norm(oi_to_pi_repeat, dim=-1, p=2)
        distance = area/vector_len

        distance_weight = 1 - torch.sigmoid(50*(distance-0.5)) # 50 0.5
        # distance_weight = 1 - torch.sigmoid(steep*(distance-0.05*H)) # 50 0.5

        epipolar_map = rearrange(distance_weight,"b (hw hw2) -> b hw hw2",hw = h*w)

        #* 如果 max(1-sigmoid) < 0.5 
        #* => min(distance) > 0.05 
        #* => 每個點離epipolar line 太遠
        #* => epipolar line 不在圖中
        #* weight map 全設為 1 
        max_values, _ = torch.max(epipolar_map, dim=-1)
        mask = max_values < 0.5
        epipolar_map[mask.unsqueeze(-1).expand_as(epipolar_map)] = 1

        return epipolar_map
    
    def iter_forward(self, dc_emb, z_emb, p,k=None,w2c=None, embeddings=None, targets=None, return_layers=False):
        
        token_embeddings_dc = dc_emb

        # add the token embedding with z_indices
        token_embeddings_z = z_emb
        token_embeddings = torch.cat([token_embeddings_dc, token_embeddings_z], 1)
        token_embeddings = token_embeddings[:, :-1, :] # remove the last one
        #! 這裡-1 作法滿神奇的
        #! 簡單來說對於 input x 來說, 在當作query 時他會把所有token 都當作往前一格
        #! 舉例來說若是要生成 rgb2, 他在query 中的位置就會是 (256+30)-1 = 285 ~ (256*2+30)-1 = 541
        #! 而若是當成key 時, 它的位置就會是正常的 286~542
        #! 這樣做的話, 假設我要生成最後一個token 827(實際上是828),他就會看到key實際 0~827的token
        #! 然後query 就會生成第 827的token, 但是實際上代表的是第828的token
        
        # drop out for teacher
        token_embeddings = self.drop(token_embeddings)
        
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        role_emb = []
        for _ in range(self.time_len-1):
            role_emb.append(self.frame_emb)
            role_emb.append(self.camera_emb)

        role_emb.append(self.frame_emb)
        role_emb = torch.cat(role_emb, 1)

        #* role emb shape (1,828,1024), 但是取前827個
        #* time emb shape (1,827,1024)
        
        role_embeddings = role_emb[:, :t, :] # each position maps to a (learnable) vector
        time_embeddings = self.time_emb[:, :t, :] # each position maps to a (learnable) vector
        
        #* x shape (B,827,1024)
        x = token_embeddings + role_embeddings + time_embeddings

        # print(f"input shape = {x.shape}")

        if return_layers:
            layers = [x]
            for block in self.blocks:
                x = block(x)
                layers.append(x)
            return layers
        
        #* 計算epipolar map [forward,backward,bidirectional,token_change]
        batch = x.shape[0]
        forward_epipolar_map = None
        backward_epipolar_map = None
        if self.epipolar!=None:
            if self.epipolar == 'forward' or self.epipolar == 'bidirectional' or self.epipolar == 'token_change' or self.epipolar == "alternately":
                # forward_epipolar_map = get_epipolar_tensor(1,h,h,k2.clone(),prev_w2c,now_w2c)
                
                w2c_0 = w2c[:,0]
                w2c_1 = w2c[:,1]
                w2c_2 = w2c[:,2]
                f01 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_0,w2c_1)
                f02 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_0,w2c_2)
                f12 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_1,w2c_2)
                forward_epipolar_map = [f01,f02,f12]
            if self.epipolar == 'backward' or self.epipolar == 'bidirectional' or self.epipolar == 'token_change' or self.epipolar == "alternately":
                # forward_epipolar_map = get_epipolar_tensor(1,h,h,k2.clone(),prev_w2c,now_w2c)
                w2c_0 = w2c[:,0]
                w2c_1 = w2c[:,1]
                w2c_2 = w2c[:,2]
                b01 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_1,w2c_0)
                b02 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_2,w2c_0)
                b12 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_2,w2c_1)
                backward_epipolar_map = [b01,b02,b12]

        # locality
        p1, p2, p3 = p
        h = self.locality(p1, p2, p3)
        # h = h.repeat(x.shape[0], 1, 1, 1)
        for block in self.blocks:
            x = block(x, h,forward_map = forward_epipolar_map,backward_map = backward_epipolar_map)
        
        # x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    
    def cross_forward(self, rgb1_emb, rgb0_emb=None,k=None,w2c=None, embeddings=None, targets=None, return_layers=False):
        # token_embeddings_dc = dc_emb

        # # add the token embedding with z_indices
        # token_embeddings_z = z_emb
        # token_embeddings = torch.cat([token_embeddings_dc, token_embeddings_z], 1)

        token_embeddings = rgb1_emb

        #! 代表使用前兩張圖片當作condition
        token_embeddings0 = None
        if rgb0_emb != None:
            token_embeddings0 = rgb0_emb
        
        # drop out for teacher
        token_embeddings = self.drop(token_embeddings)
        
        b = token_embeddings.shape[0]
        t = self.start_token.shape[1]
        start_token = self.start_token.expand(b, -1, -1)
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        role_emb = []
        for _ in range(1):
            role_emb.append(self.frame_emb)
            role_emb.append(self.camera_emb)

        # role_emb.append(self.frame_emb)
        role_emb = torch.cat(role_emb, 1)

        #* role emb shape (1,286,1024)
        #* time emb shape (1,286,1024)
        
        role_embeddings = role_emb[:, :t, :] # each position maps to a (learnable) vector
        time_embeddings = self.time_emb[:, :t, :] # each position maps to a (learnable) vector

        
        #* x shape (B,286,1024)
        x = start_token + role_embeddings + time_embeddings

        #! test position emb for src img
        if self.sep_pe==False:
            token_embeddings = token_embeddings + role_embeddings + time_embeddings
        else: #* 分開做position encode
            role_emb2 = []
            for _ in range(1):
                role_emb2.append(self.frame_emb2)
                role_emb2.append(self.camera_emb2)

            # role_emb.append(self.frame_emb)
            role_emb2 = torch.cat(role_emb2, 1)
            role_embeddings2 = role_emb2[:, :t, :] # each position maps to a (learnable) vector
            time_embeddings2 = self.time_emb2[:, :t, :] # each position maps to a (learnable) vector

            token_embeddings = token_embeddings + role_embeddings2 + time_embeddings2

            if token_embeddings0 != None:
                token_embeddings0 = token_embeddings0 + role_embeddings2 + time_embeddings2

        # print(f"input shape = {x.shape}")

        if return_layers:
            layers = [x]
            for block in self.blocks:
                x = block(x)
                layers.append(x)
            return layers
        

        #* 計算epipolar map [forward,backward,bidirectional,token_change]
        batch = x.shape[0]
        forward_epipolar_map = None
        backward_epipolar_map = None
        if self.epipolar!=None:
            if self.epipolar == 'forward' or self.epipolar == 'bidirectional' or self.epipolar == 'token_change' or self.epipolar == "alternately":
                  
                w2c_0 = w2c[:,0]
                w2c_1 = w2c[:,1]
                f01 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_0,w2c_1)

                forward_epipolar_map = [f01]

                if rgb0_emb!=None:
                    w2c_2 = w2c[:,2]
                    f02 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_0,w2c_2)
                    f12 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_1,w2c_2)
                    forward_epipolar_map = [f02,f12]
            if self.epipolar == 'backward' or self.epipolar == 'bidirectional' or self.epipolar == 'token_change' or self.epipolar == "alternately":
              
                w2c_0 = w2c[:,0]
                w2c_1 = w2c[:,1]
                b01 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_1,w2c_0)
                backward_epipolar_map = [b01]

                if rgb0_emb!=None:
                    w2c_2 = w2c[:,2]
                    b02 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_2,w2c_0)
                    b12 = self.get_epipolar_tensor(batch,16,16,k.clone(),w2c_2,w2c_1)
                    backward_epipolar_map = [b02,b12]
        # locality
        # p1, p2, p3 = p
        # h = self.locality(p1, p2, p3)
        # h = h.repeat(x.shape[0], 1, 1, 1)
        iteration = 0
        for block in self.blocks:
            if iteration%2==0:
                #* cross
                x = block(x,token_embeddings,forward_map = forward_epipolar_map,backward_map = backward_epipolar_map,src_encode0=token_embeddings0)
            elif iteration%2==1:
                #* self
                x = block(x,x,forward_map = forward_epipolar_map,backward_map = backward_epipolar_map)
            iteration += 1
        
        # x_forward = self.ln_f(x)
        # logits_forward = self.head(x_forward)

        #! 作林老師說的reconstruction 
        iteration = 0
        for block in self.blocks2:
            if iteration%2==0:
                #* cross
                x = block(x,token_embeddings,forward_map = forward_epipolar_map,backward_map = backward_epipolar_map,src_encode0=token_embeddings0)
            elif iteration%2==1:
                #* self
                x = block(x,x,forward_map = forward_epipolar_map,backward_map = backward_epipolar_map)
            iteration += 1
        
        # x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        # return logits, loss, logits_forward
        return logits, loss
    
    def test(self, dc_emb, z_indices, p,forward_epipolar_map=None,backward_epipolar_map=None, embeddings=None, targets=None, return_layers=False, return_bias=False):
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

        #* 計算epipolar map [forward,backward,bidirectional,token_change]
        # batch = x.shape[0]
        # forward_epipolar_map = None
        # backward_epipolar_map = None
        # if self.epipolar!=None:
        #     if self.epipolar == 'forward' or self.epipolar == 'bidirectional' or self.epipolar == 'token_change':
        #         # forward_epipolar_map = get_epipolar_tensor(1,h,h,k2.clone(),prev_w2c,now_w2c)
                
        #         w2c_0 = w2c[:,0]
        #         w2c_1 = w2c[:,1]
        #         w2c_2 = w2c[:,2]
        #         f01 = get_epipolar_tensor(batch,16,16,k.clone(),w2c_0,w2c_1)
        #         f02 = get_epipolar_tensor(batch,16,16,k.clone(),w2c_0,w2c_2)
        #         f12 = get_epipolar_tensor(batch,16,16,k.clone(),w2c_1,w2c_2)
        #         forward_epipolar_map = [f01,f02,f12]
        #     if self.epipolar == 'backward' or self.epipolar == 'bidirectional' or self.epipolar == 'token_change':
        #         # forward_epipolar_map = get_epipolar_tensor(1,h,h,k2.clone(),prev_w2c,now_w2c)
        #         w2c_0 = w2c[:,0]
        #         w2c_1 = w2c[:,1]
        #         w2c_2 = w2c[:,2]
        #         b01 = get_epipolar_tensor(batch,16,16,k.clone(),w2c_1,w2c_0)
        #         b02 = get_epipolar_tensor(batch,16,16,k.clone(),w2c_2,w2c_0)
        #         b12 = get_epipolar_tensor(batch,16,16,k.clone(),w2c_2,w2c_1)
        #         backward_epipolar_map = [b01,b02,b12]

        # locality
        p1, p2, p3 = p
        h = self.locality(p1, p2, p3)
        h = h.repeat(x.shape[0], 1, 1, 1) #* (1,1,827,827)
        
        #* x shape (1,286,1024)
        #* f01 shape (1,256,256)
        # for block in self.blocks:
        #     x = block(x, h)
        for block in self.blocks:
            x = block(x, h,forward_map = forward_epipolar_map,backward_map = backward_epipolar_map)
            
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
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