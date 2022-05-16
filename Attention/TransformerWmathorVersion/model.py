import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor as Tensor
import numpy as np
"""
需要实现以下几个部分
1. 位置编码
2. PaddingMask
3. SequenceMask
4. ScaledDotProductAttention
5. Multi-HeadAttention
6. FeedForward NN
7. Encoder Layer
8. Decoder Layer
9. Encoder and Decoder
10. Transformer
"""

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, drop_out=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self. dropout = nn.Dropout(drop_out)
        pe = t.zeros((max_len, d_model))
        position = t.arange(0, max_len, dtype=t.float32).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2).float()
                         * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(position*div_term)
        pe[:, 1::2] = t.cos(position*div_term)
        # pe.shape = (max_len,d_model)
        # unsqueeze = (1,max_len,d_model)
        # transform = (max_len,1,d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        X.shape is (seq_len,batch_size,d_model)
        利用广播机制对于每个batch加上位置编码
        """
        # 直接加上位置编码就可以辣
        x = x+self.pe[:x.size(0), :]
        return self.dropout.forward(x)


def get_attn_pad_mask(seq_q: Tensor, seq_k: Tensor,):
    """
    seq_q.shape = (batch_size,seq_len)\n
    seq_k.shape = (batch_size,seq_len)\n
    seq_len可以src或者是tgt的\n
    在seq_q或者是seq_k中seq_len可能是不相等的\n
    """
    batch_size_q, len_q = seq_q.shape
    batch_size_k, len_k = seq_k.shape
    # eq(zero) is PAD token
    # pad_atten_mask.shap is [batch_size,1,len_k]
    # 元素为True代表有掩膜
    # 这一步是这个函数的核心操作
    pad_atten_mask = seq_k.data.eq(0).unsqueeze(1)
    # 增广一下
    return pad_atten_mask.expand(batch_size_k, len_q, len_k)


def get_attn_subsequence_mask(seq: Tensor):
    """
    seq.shape = (batch_size,seq_len)\n
    在Decoder中需要用到，使用它来屏蔽未来时刻的单词信息
    """
    atten_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 上三角矩阵
    subsequence_mask = np.triu(np.ones(atten_shape), k=1)
    # 转Tensor
    subsequence_mask = t.from_numpy(subsequence_mask).byte()
    # subsequence_mask.shape = (batch_size,seq_len,seq_len)
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, attn_mask: Tensor):
        """
        Q.shape = (batch_size,num_heads,len_q,d_k)
        K.shape = (batch_size,num_heads,len_k,d_k)
        V.shape = (batch_size,num_heads,len_v(=len_k),d_v)
        attn_mask.shape = (batch_size,num_heads,seq_len,seq_len)
        """
        # 进行内积并使用负无穷填充遮蔽区域
        # scores.shape = [batch_size,num_heads,len_q,len_k]
        scores = t.matmul(Q, K.transpose(-1, -2))/np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = t.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    在编码器中、解码器遮蔽自注意力中，编码器-解码器注意力中，我们会用到这些东西\n
    具体地，在编码器中传入的QKV都是相等的，是enc_inputs
    在解码器自注意力中传入的QKV也是相等的，是dec_inputs
    在编码器-解码器注意力中，就不一样了，QKV分别是dec_outputs,enc_outputs,enc_outputs
    """

    def __init__(self,):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        # 使用线性层将它们的维度降下来
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)

    def forward(self, input_Q: Tensor, input_K: Tensor, input_V: Tensor, attn_mask: Tensor):
        """
        Q.shape = (batch_size,len_q,d_k)
        K.shape = (batch_size,len_k,d_k)
        V.shape = (batch_size,len_v(=len_k),d_v)
        attn_mask.shape = (batch_size,seq_len,seq_len)
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 投影->切片->转置
        Q = self.W_Q.forward(input_Q).view(
            batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K.forward(input_K).view(
            batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V.forward(input_V).view(
            batch_size, -1, n_heads, d_v).transpose(1, 2)
        # 获取注意力遮罩,对于每个注意力头我们都有一样的遮罩，所以需要复制
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # 获取Attention结果
        # context.shape = (batch_size,n_heads,len_q,d_v)
        # attn.shape = (batch_size,n_heads,len_q,len_k)
        context, attn = ScaledDotProductAttention().forward(Q, K, V, attn_mask)
        # 还原维度
        context = context.reshape(batch_size, -1, n_heads*d_v)
        output = self.fc.forward(context)
        return F.layer_norm(output+residual, (d_model,)), attn


class FeedForwardNet(nn.Module):
    """
    简单的线性层加上残差连接和LayerNorm
    """

    def __init__(self,):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, inputs: Tensor):
        """
        inputs.shape = (batch_size,seq_length,d_model)
        """
        residual = inputs
        output = self.fc.forward(inputs)
        return F.layer_norm(output+residual, (d_model,))


class EncodreLayer(nn.Module):
    """
    一个编码器层，它包含一个多头自注意力模块和一个前馈神经网络，注意里面有LayerNorm和残差连接
    """

    def __init__(self,):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.ffn = FeedForwardNet()

    def forward(self, enc_inputs: Tensor, enc_self_attn_mask: Tensor):
        """
        enc_inputs.shape = (batch_size,src_len,d_model)
        enc_self_attn_mask.shape = (batch_size,src_len,src_len)
        """
        # 在encoder中，QKV都是编码器输入
        # enc_outputs.shape = (batch_size,src_len,d_model)
        # attn.shape= (batch_size,n_heads,src_len,src_len)
        enc_outputs, attn = self.enc_self_attn.forward(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.ffn.forward(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    """
    编码器，它包含一嵌入层（负责将输入语句转换成嵌入向量），一个位置编码，还有n个编码器层
    """

    def __init__(self, src_vocab_size):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncodreLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs: Tensor):
        """
        enc_inputs.shape = (batch_size,src_len)
        """
        # enc_outputs.shape = (batch_size,src_len,d_model)
        enc_outputs = self.src_emb.forward(enc_inputs)
        # 转维度之后再丢进位置编码中便于并行化操作 pos_emb的输入shape(seq_len,batch_size,d_model)
        # 后面的一个transpose是为了还原 shape (batch_size,seq_len,d_model)
        enc_outputs = self.pos_emb.forward(
            enc_outputs.transpose(0, 1)).transpose(1, 0)
        # 遇到pad这种词元需要就要设置Mask
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        # 将注意力保存下来
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs.shape = (batch_size,src_len,d_model)
            # enc_self_attn.shape = (batch_size,n_heads,src_len,src_len)
            enc_outputs, enc_self_attn = layer.forward(
                enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    """
    解码器的层，注意到它的MHA输入包含了编码器的输出
    """

    def __init__(self,):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.ffn = FeedForwardNet()

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor, dec_self_attn_mask: Tensor, dec_enc_attn_mask: Tensor):
        """
        dec_inputs.shape = (batch_size,tgt_len,d_model)
        enc_outputs.shape = (batch_size,src_len,d_model)
        dec_self_attn_mask.shape = (batch_size,tgt_len,tgt_len)
        dec_enc_attn_mask.shape = (batch_size, tgt_len,src_len)
        """
        # 自注意力
        dec_outputs, dec_self_attn = self.dec_self_attn.forward(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # 编码器解码器注意力
        dec_outputs, dec_enc_attn = self.dec_enc_attn.forward(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # 通过FFN
        dec_outputs = self.ffn.forward(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    """
    解码器架构与编码器大同小异
    ，不同的是有两个地方的attention输入不一样
    ，而且为了防止GT泄露进行上三角矩阵的遮罩
    """

    def __init__(self, tgt_vocab_size):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self. layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs: Tensor, enc_inputs: Tensor, enc_outpus: Tensor):
        """
        dec_inputs.shape = (batch_size,tgt_len)
        enc_inputs.shape = (batch_size,src_len)
        enc_outputs.shape = (batch_size,src_len,d_model)
        """
        dec_outputs = self.tgt_emb.forward(dec_inputs)
        dec_outputs = self.pos_emb.forward(
            dec_outputs.transpose(0, 1)).transpose(1, 0)
        # 求selfAttn的mask，这里要加上子序列的mask防止GT暴露
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = t.gt(
            (dec_self_attn_pad_mask+dec_self_attn_subsequence_mask), 0)
        # 对于dec和enc求一个mask
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer.forward(
                dec_outputs, enc_outpus, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.encoder = Encoder(src_vocab_size)
        self.decoder = Decoder(tgt_vocab_size)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs: Tensor, dec_inputs: Tensor):
        """
        enc_inputs.shape = (batch_size,src_len)
        dec_inputs.shape = (batch_size,tgt_len)
        """
        enc_outputs, enc_self_attns = self.encoder.forward(enc_inputs)

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder.forward(
            dec_inputs, enc_inputs, enc_outputs)

        dec_logits = self.projection.forward(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
