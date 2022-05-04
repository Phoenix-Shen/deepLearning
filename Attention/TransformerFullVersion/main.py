from tensorboardX import SummaryWriter
from datasets import load_data_nmt
from model import TransformerEncoder, TransformerDecoder, EncoderDecoder
import yaml
import torch as t
from utils import *


if __name__ == "__main__":
    # Hyper Parameterss
    with open("./Attention/TransformerFullVersion/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, yaml.FullLoader)
    writer = SummaryWriter("./Attention/TransformerFullVersion/logs")
    num_hiddens, num_layers, dropout, batch_size, num_steps\
        = config["num_hiddens"], config["num_layers"], config["dropout"], config["batch_size"], config["num_steps"]
    lr, num_epochs, device = config["lr"], config["num_epochs"], t.device(
        "cuda" if config["use_gpu"] else "cpu")
    ffn_num_input, ffn_num_hiddens, num_heads = config[
        "ffn_num_input"], config["ffn_num_hiddens"], config["num_heads"]
    key_size, query_size, value_size = config["key_size"], config["query_size"], config["value_size"]
    norm_shape = config["norm_shape"]
    # Prepare Train Data
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    # Construct Model
    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout,
    )
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout,
    )

    net = EncoderDecoder(encoder, decoder)
    net.train()
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = t.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    # Start Training
    for epoch in range(num_epochs):
        for batch in train_iter:
            optimizer.zero_grad()
            # 解包数据，转cuda
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 获取开始符号的下标
            bos = t.tensor([tgt_vocab["<bos>"]]*Y.shape[0],
                           device=device).reshape(-1, 1)
            # 在每个句子之前加上开始符号
            dec_input = t.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = net.forward(X, dec_input, X_valid_len)
            l = loss.forward(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
        with t.no_grad():
            writer.add_scalar("loss", l.sum()/num_tokens,)
    print(f'loss {l.sum()/num_tokens:.3f}')

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {bleu(translation, fra, k=2):.3f}')
