import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    在处理词元序列的时候，自注意力因为并行计算放弃了顺序操作，为了使用序列的顺序信息，在输入表示中添加位置编码来注入绝对或者是相对的位置信息。
    """

    def __init__(self, num_hiddens: int, dropout: float, max_len=1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的p， p.shape = 1, num_steps, num_hiddens
        self.p = t.zeros((1, max_len, num_hiddens))
        # 先计算出来下标
        # 先定义出一个长度为max_len的数组，这就是公式里面的i
        I = t.arange(0, max_len, 1, dtype=t.float32).reshape(-1, 1)
        # 计算分式下面的值
        # EXP = 2j/d
        EXP = t.arange(0, num_hiddens, 2, dtype=t.float32)/num_hiddens
        # 计算最终的值
        V = I / t.pow(10000, EXP)

        # 给位置赋sin 和 cos值
        # 注意这里是序列切片，序列切片的操作是,[开始：结束：步长]，任何开始结束或者步长都可以被丢弃
        # 奇数位赋cos，偶数位赋sin
        self.p[:, :, 0::2] = t.sin(V)
        self.p[:, :, 1::2] = t.cos(V)

    def forward(self, X: Tensor) -> Tensor:
        """
        给序列附加位置编码
        """
        X = X+self.p[:, :X.shape[1], :].to(X.device)
        X = self.dropout.forward(X)
        return X


def sequence_mask(X: Tensor, valid_len: Tensor, value=0) -> Tensor:
    """
    在序列中屏蔽不相关的项
    """
    # x.shape = [batch,step,hidden]
    maxlen = X.size(1)
    # None 简单说就是说它增加了一个维度
    mask = t.arange((maxlen), dtype=t.float32, device=X.device)[
        None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X: Tensor, valid_lens: Tensor) -> Tensor:
    """
    如果一个句子长度不满足我们给定的长度，我们就进行填充，为了仅仅将有意义的词元作为值来获取注意力汇聚\n
    我们要指定一个有效序列长度(即词元的个数),以便在softmax计算的时候过滤掉指定范围的位置，这个函数\n
    的作用就是将任何超出有效长度的位置都被遮蔽然后设置为0
    """
    # 如果没有长度限制，直接返回softmax结果，是对最后一维进行Softmax操作
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 如果维度是1的话，那么就直接复制
            valid_lens = t.repeat_interleave(valid_lens, shape[1])
        else:
            # 否则压缩到一个维度上面去
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被遮蔽的元素使用非常大的负值来替换掉，使softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        # 使用负数填充之后，再进行最后一个维度的Softmax操作
        return F.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """
    缩放点积注意力，它要求查询和键都有一样的形状，因为要进行矩阵相乘，直接将查询和键做内积
    """

    def __init__(self, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens=None) -> Tensor:
        """
        根据键和查询来返回相应的值，注意力分数是查询和键的相似度，注意力权重是分数的softmax结果
        """
        # 获得特征维度
        d = queries.shape[-1]
        # 查询和键的转置进行矩阵相乘，然后除以特征维度进行归一化
        scores = t.bmm(queries, keys.transpose(1, 2))/math.sqrt(d)
        # 进行softmax操作得到注意力权重，注意力权重的和是1
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 与输入的值进行矩阵相乘，得到最后的attention结果
        attention_value = t.bmm(self.dropout.forward(
            self.attention_weights), values)
        return attention_value


def transpose_qkv(X: Tensor, num_heads: int) -> Tensor:
    """
    为了多头注意力的并行而转换维度
    """
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出 (batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X: Tensor, num_heads: int) -> Tensor:
    """
    逆转transpose_qkv的操作
    """
    # 将输出扩展维度
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # Transpose
    X = X.permute(0, 2, 1, 3)
    # 然后再进行还原
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """
    多头注意力，里面有好多个DotProductAttention，而且实现了并行计算
    """

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens: Tensor) -> Tensor:
        """
        将多头注意力并到batch维度实现并行化计算
        """
        queries = transpose_qkv(self.W_q.forward(queries), self.num_heads)
        keys = transpose_qkv(self.W_k.forward(keys), self.num_heads)
        values = transpose_qkv(self.W_v.forward(values), self.num_heads)
        # 将valid_lens复制num_head份然后进行并行处理
        if valid_lens is not None:
            valid_lens = t.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention.forward(queries, keys, values, valid_lens)
        # output_concat 的形状(batch_size, num_queries,num_hiddens)
        output = transpose_output(output, self.num_heads)
        output = self.W_o.forward(output)
        return output


class PositionWiseFFN(nn.Module):
    """
    定义基于位置的前馈网络，其实就是MLP\n
    但是线性层会将最后一维除外的维度都看做batch
    """

    def __init__(self, ffn_num_input: int, ffn_num_hiddens: int, ffn_num_outputs: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X: Tensor) -> Tensor:
        """
        前馈网络的forward，只改变最后一维的size，它是ffn_num_outputs
        """
        X = self.dense1.forward(X)
        X = F.relu(X)
        X = self.dense2.forward(X)
        return X


class AddNorm(nn.Module):
    """
    定义残差连接和层归一化函数
    """

    def __init__(self, normalized_shape, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        进行残差连接之后再进行层归一化
        """
        X = self.dropout.forward(Y)+X
        X = self.ln.forward(X)
        return X


class EncoderBlock(nn.Module):
    """
    Transformer 编码器块，\n
    由一个多头注意力2个Norm模块和一个基于位置的前馈网络组成\n
    一个编码器中有多个这种玩意
    """

    def __init__(self, key_size: int, query_size: int,
                 value_size: int, num_hiddens: int, norm_shape,
                 ffn_num_input: int, ffn_num_hiddens: int, num_heads: int,
                 dropout: int, use_bias=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X: Tensor, valid_lens: Tensor) -> Tensor:
        Y = self.attention.forward(X, X, X, valid_lens)
        Y = self.addnorm1.forward(X, Y)
        Y = self.addnorm2.forward(Y, self.ffn.forward(Y))
        return Y


class TransformerEncoder(nn.Module):
    """
    Transformer的编码器，它包含若干个EncoderBlock\n
    inputs: the sequential data with shape (batch,num_steps)
    outputs: the processed data with shape (batch,num_steps,num_hiddens)
    """

    def __init__(self, vocab_size, key_size, query_size,
                 value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 use_bias=False, **kwargs) -> None:
        super().__init__(**kwargs)
        # 将参数保存到成员变量里面去
        self.num_hiddens = num_hiddens
        # 将vocab转换成嵌入向量
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 还要附加上位置编码
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout, 1000)
        # 然后使用nn.Sequential 进行多个编码器块的叠加
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 EncoderBlock(key_size, query_size,
                                              value_size, num_hiddens, norm_shape,
                                              ffn_num_input, ffn_num_hiddens, num_heads,
                                              dropout, use_bias)
                                 )

    def forward(self, X: Tensor, valid_lens: Tensor, *args) -> Tensor:
        # 与根号num_hiddens相乘，避免位置编码过大导致只学习位置编码
        X = self.pos_encoding.forward(
            self.embedding.forward(X)*math.sqrt(self.num_hiddens))
        # 将Attention权重先定义，等会进行赋值，方便可视化操作
        self.attention_weights = [None]*len(self.blks)

        # 逐级进行前向传播，拿出Attention
        for i, blk in enumerate(self.blks):
            X = blk.forward(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights

        return X


class DecoderBlock(nn.Module):
    """
    解码器也是由多个相同的层组成，每个层都包含了三个子层：
    解码器自注意力、残差连接和层归一化、基于位置的前馈网络
    """

    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, i, **kwargs) -> None:
        super().__init__(**kwargs)
        # 第i个解码器块
        self.i = i
        # 主要成分是 3个AddNorm 2个MultiheadAttention，其中有一个是Masked，1个FFN
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X: Tensor, state: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = t.cat((state[2][self.i], X), dim=1)
        # concatenation比单纯的相加更能保留数据
        state[2][self.i] = key_values

        # 在训练过程中，将后面的内容遮蔽起来，但是在预测过程中就不用了，因为我们也看不到后面的东西
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = t.arange(
                1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 进行前向传播操作
        # 第一个attention模块用的是target input作为kqv
        X2 = self.attention1.forward(
            X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1.forward(X, X2)
        # 第二个attention使用Y作为q，编码器的输出作为kv
        Y2 = self.attention2.forward(
            Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2.forward(Y, Y2)
        # 经过前馈网络之后返回结果
        return self.addnorm3.forward(Z, self.ffn.forward(Z)), state


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs) -> None:
        super().__init__(**kwargs)
        # 保存参数到成员变量
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        # 将vocab转成嵌入向量
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 添加位置编码
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout, 1000)
        # 定义解码器块
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlock(key_size, query_size, value_size,
                                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))
        # 要输出，所以还要定义一个全连接层进行输出
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs: Tensor, enc_validlens: Tensor, *args):
        return [enc_outputs, enc_validlens, [None]*self.num_layers]

    def forward(self, X: Tensor, state: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        # 如法炮制
        X = self.pos_encoding.forward(
            self.embedding.forward(X)*math.sqrt(self.num_hiddens))
        # 要提取权重进行可视化，这里较为啰嗦
        # 有两个多头注意力，所以就要in range(2)
        self._attention_weights = [[None]*len(self.blks) for _ in range(2)]

        # 传入到DecoderBlock中
        for i, blk in enumerate(self.blks):
            X, state = blk.forward(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # "编码器-解码器"自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        # 返回全连接层的输出和当前的状态
        return self.dense.forward(X), state

    @property
    def attention_weights(self):
        """
        将注意力权重定义为属性，以便快速访问。
        """
        return self._attention_weights


class EncoderDecoder(nn.Module):
    """
    定义编码器和解码器抽象类，其中编码器和解码器可以是任何类型的
    """

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X: Tensor, dec_X: Tensor, *args) -> tuple[Tensor, Tensor]:
        """
        前向传播函数
        """
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder.forward(dec_X, dec_state)


# %% TEST
if __name__ == "__main__":
    encoder = TransformerEncoder(
        200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval()
    output = encoder.forward(t.ones((2, 100), dtype=t.long),
                             valid_lens=t.tensor([3, 2]))

    print(output.shape)

    decoder = TransformerDecoder(
        200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5
    )
    decoder.eval()
    state = decoder.init_state(output, t.tensor([3, 2]))
    output_dec = decoder.forward(t.ones((2, 100), dtype=t.long), state)
    print(output_dec[0].shape)
