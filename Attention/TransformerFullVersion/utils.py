import torch as t
import torch.nn as nn
from torch import Tensor
from datasets import truncate_pad
import math
import collections


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = t.sqrt(sum(t.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta/norm


def sequence_mask(X: Tensor, valid_len: Tensor, value=0):
    """
    在序列中遮蔽不相关的项，防止它们影响到损失计算
    """
    maxlen = X.size(1)
    sequen_data = t.arange((maxlen), dtype=t.float32, device=X.device)
    mask = sequen_data[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    带有遮蔽的交叉熵损失
    """

    def forward(self, pred: Tensor, label: Tensor, valid_len: Tensor):
        # pred.shape = batch,step,vsize
        # label.shape = batch,step
        # valid_len.shape = batch
        weights = t.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        pred = pred.permute(0, 2, 1)
        unweighted_loss = super().forward(pred, label)
        weighted_loss = (unweighted_loss*weights).mean(dim=1)
        return weighted_loss


def xavier_init_weights(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测
    Defined in :numref:`sec_seq2seq_training`"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = t.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = t.unsqueeze(
        t.tensor(src_tokens, dtype=t.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = t.unsqueeze(t.tensor(
        [tgt_vocab['<bos>']], dtype=t.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(t.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):  # @save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
