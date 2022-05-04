# %%
import collections
import torch as t
from torch.utils import data


def read_data_nmt():
    """
    载入英语-法语数据集
    """
    with open("dataset/fra-eng/fra.txt", "r", encoding="utf-8") as f:
        return f.read()


def preprocess_nmt(text: str):
    """
    预处理英语-法语数据集，
    使用空格替换不间断的空格，
    使用小写字母替换大写字母，
    在单词和标点符号之间加上空格，
    """
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != " "
    # 使用空格来替换不间断的空格
    # 使用小写字母来替换大写字母
    text = text.replace("\u202f", " ").replace("\xa0", " ").lower()
    # 在单词和标点之间插入空格
    out = [" " + char if i >
           0 and no_space(char, text[i-1]) else char for i, char in enumerate(text)]
    return "".join(out)


def tokenize_nmt(text: str, num_examples=None):
    """
    词元化英语-法语数据集
    """
    source, target = [], []

    for i, line in enumerate(text.split("\n")):
        # 到达了指定长度 则直接跳出循环
        if num_examples and i >= num_examples:
            break
        # 以分隔符分离英语和法语
        parts = line.split("\t")
        # 如果长度是2（排除不合理的长度）
        if len(parts) == 2:
            source.append(parts[0].split(" "))
            target.append(parts[1].split(" "))
    return source, target

# 词表


def count_corpus(tokens: list):
    # token 是1D列表或者是2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """
    文本词表
    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现的频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(
            counter.items(), key=lambda x: x[1], reverse=True)

        # 未知词元的索引为0
        self.idx_to_token = ["<unk>"]+reserved_tokens
        self.token_to_idx = {token: idx for idx,
                             token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token)-1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freaqs(self):
        return self._token_freqs


def truncate_pad(line, num_steps, padding_token):
    """
    截断或者是填充文本序列
    """
    if len(line) > num_steps:
        return line[:num_steps]  # truncate
    return line + [padding_token]*(num_steps-len(line))


def build_array_nmt(lines, vocab: Vocab, num_steps):
    """
    将机器翻译的文本序列转换成小批量
    """
    lines = [vocab[l] for l in lines]
    lines = [l+[vocab["<eos>"]] for l in lines]

    array = t.tensor([truncate_pad(l, num_steps, vocab["<pad>"])
                     for l in lines])
    valid_len = (array != vocab["<pad>"]).type(t.int32)
    valid_len = valid_len.sum(1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器
    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """
    返回翻译数据集的迭代器和词表
    """
    raw_data = read_data_nmt()
    text = preprocess_nmt(raw_data)
    source, target = tokenize_nmt(text, num_examples)

    src_vocab = Vocab(source, 2, ["<pad>", "<bos>", "<eos>"])
    tgt_vocab = Vocab(target, 2, ["<pad>", "<bos>", "<eos>"])

    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


# %% TEST
if __name__ == "__main__":
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(t.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(t.int32))
        print('Y的有效长度:', Y_valid_len)
        break
