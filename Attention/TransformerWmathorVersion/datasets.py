import torch as t
import torch.utils.data as data
from torch import Tensor

"""
简单的一个德语-英语数据集
只有两个句子，而且手动添加了起止符号和Padding
S 代表句子的开头
E 代表句子的结束
P 代表长度不足的时候进行填充
"""

sentences = [
    ["ich mochte ein bier P", "S i want a beer .", "i want a beer . E"],
    ["ich mochte ein cola P", "S i want a coke .", "i want a coke . E"],
]

# padding 应该是0
src_vocab = {"P": 0, "ich": 1, "mochte": 2, "ein": 3, "bier": 4, "cola": 5, }
src_vocab_size = len(src_vocab)

tgt_vocab = {"P": 0, "i": 1, "want": 2, "a": 3,
             "beer": 4, "coke": 5, "S": 6, "E": 7, ".": 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

# max_src_sequencelength = 5,max_tgt_sequencelength=6
src_len, tgt_len = 5, 6


"""
在这里构造数据，其实我们的数据都已经做了padding
然后需要创建一个[batch_size=2,len=5或者6(取决于是英语还是德语)]的数据集
"""


def make_data(sentences: list[str]) -> tuple[Tensor, Tensor, Tensor]:
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split(" ")]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split(" ")]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split(" ")]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return t.LongTensor(enc_inputs), t.LongTensor(dec_inputs), t.LongTensor(dec_outputs)


# 声明数据
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

# 定义数据集


class Mydataset(data.Dataset):
    def __init__(self, enc_inputs: Tensor, dec_inputs: Tensor, dec_outputs: Tensor) -> None:
        super(Mydataset, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self,):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = data.DataLoader(
    Mydataset(enc_inputs, dec_inputs, dec_outputs), 2, True)
