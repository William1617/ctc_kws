
"""Encoder definition."""

import torch
import torch.nn as nn
from typing import List, Tuple, Union,Optional

from modules import MultiHeadedAttention,Conv2dSubsampling4,ConvolutionalGatingMLP,PositionalEncoding

from Processor import make_pad_mask

class BranchformerEncoderLayer(torch.nn.Module):


    def __init__(
        self,
        size: int,
        attn: Optional[torch.nn.Module],
        cgmlp: Optional[torch.nn.Module],
        dropout_rate: float,
        merge_method: str,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp
        self.merge_method = merge_method
        
        self.stochastic_depth_rate = stochastic_depth_rate
    
        self.norm_mha = nn.LayerNorm(size)  # for the MHA module

        self.norm_mlp = nn.LayerNorm(size)  # for the MLP module
        self.norm_final = nn.LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

        # # attention-based pooling for two branches
        self.pooling_proj1 = torch.nn.Linear(size, 1)
        self.pooling_proj2 = torch.nn.Linear(size, 1)

        # # linear projections for calculating merging weights
        self.weight_proj1 = torch.nn.Linear(size, 1)
        self.weight_proj2 = torch.nn.Linear(size, 1)

        if self.merge_method == "concat":
            self.merge_proj = torch.nn.Linear(size + size, size)

        elif self.merge_method == "learned_ave":
                # linear projection after weighted average
            self.merge_proj = torch.nn.Linear(size, size)

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) :

        stoch_layer_coeff = 1.0
       
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        if self.training and self.stochastic_depth_rate > 0:
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        # Two branches
        x1 = x
        x2 = x

        x1 = self.norm_mha(x1)
        x_att = self.attn(x1, x1, x1, mask_pad)
        x1 = self.dropout(x_att)

        x2 = self.norm_mlp(x2)
        x2 = self.cgmlp(x2)
        x2 = self.dropout(x2)

        if self.merge_method == "concat":
            x = x + stoch_layer_coeff * self.dropout(self.merge_proj(torch.cat([x1, x2], dim=-1)))
        elif self.merge_method == "learned_ave":
            score1 = (self.pooling_proj1(x1).transpose(1, 2) / self.size**0.5)
            if(mask_pad.size(2)>0):
                score1 = score1.masked_fill(mask_pad.eq(0), -float('inf'))
                score1 = torch.softmax(score1, dim=-1).masked_fill(mask_pad.eq(0), 0.0)
            else:
                score1=torch.softmax(score1,dim=-1)

            pooled1 = torch.matmul(score1, x1).squeeze(1)  # (batch, size)
            weight1 = self.weight_proj1(pooled1)  # (batch, 1)

                    # branch2
            score2 = (self.pooling_proj2(x2).transpose(1, 2) / self.size**0.5)
            if(mask_pad.size(2)>0):
                score2 = score2.masked_fill(mask_pad.eq(0), -float('inf'))
                score2 = torch.softmax(score2, dim=-1).masked_fill(mask_pad.eq(0), 0.0)
            else:
                score2 = torch.softmax(score2, dim=-1)

            pooled2 = torch.matmul(score2, x2).squeeze(1)  # (batch, size)
            weight2 = self.weight_proj2(pooled2)  # (batch, 1)

                    # normalize weights of two branches
            merge_weights = torch.softmax(
                        torch.cat([weight1, weight2], dim=-1), dim=-1)  
            # (batch, 2)
            merge_weights = merge_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, 2, 1, 1)
            w1, w2 = merge_weights[:, 0], merge_weights[:, 1]  # (batch, 1, 1)

            x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(w1 * x1 + w2 * x2) )
           
        else:
            raise RuntimeError(f"unknown merge method: {self.merge_method}")
       

        x = self.norm_final(x)

        return x



class Branchformer(nn.Module):
    """Branchformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        vocab_size: int,
    
        attention_heads: int = 4,
        cgmlp_linear_units: int = 200,
        cgmlp_conv_kernel: int = 9,

        merge_method: str = "learned_ave",
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        train_flag: bool = False,
       # global_cmvn: torch.nn.Module = None,
      
    ):
        super().__init__()
        #  self.cmvn=GlobalCMVN(mean,istd,True)

        self.subsample = Conv2dSubsampling4(input_size,output_size)
        self.emb=PositionalEncoding(output_size,positional_dropout_rate)

        encoder_selfattn_layer = MultiHeadedAttention(attention_heads,
                output_size,attention_dropout_rate,)
       
        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate)

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks
        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.encoders = torch.nn.ModuleList([BranchformerEncoderLayer(
            output_size,
            encoder_selfattn_layer,
           
            cgmlp_layer(*cgmlp_layer_args) ,
            dropout_rate,
            merge_method,
            stochastic_depth_rate[lnum]) for lnum in range(num_blocks)
        ])
        self.after_norm = nn.LayerNorm(output_size)
        self.final_lineae=nn.Linear(output_size,vocab_size)
        # self.global_cmvn = global_cmvn
    
        self.trainflag=train_flag
      

    def forward(
        self,
        xs: torch.Tensor,
        xs_len:torch.Tensor = torch.ones((0),dtype=torch.int)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if(self.trainflag):
            mask=~make_pad_mask(xs_len,xs.size(1)).unsqueeze(1)
        else:
            mask=torch.ones((0, 0, 0), dtype=torch.bool)
      #  if self.global_cmvn is not None:
       #     xs = self.global_cmvn(xs)
        xs,masks=self.subsample(xs,mask)
        xs, pos_emb = self.emb(xs)
        mask_pad = masks  # (B, 1, T/subsample_rate)

        for layer in self.encoders:
            xs = layer(xs, mask_pad)

        xs = self.after_norm(xs)
      
        return xs, masks
if __name__=='__main__':
    transformer=Branchformer(80,100,45,num_blocks=3,trainflag=True)
    dummy_input=torch.ones(3,100,80)
    xs_len=torch.ones(3)*100
    xs_len[0]=86
    xs_len[2]=80
    dummy_output,mask=transformer(dummy_input,xs_len)
    print(dummy_output.shape)

    