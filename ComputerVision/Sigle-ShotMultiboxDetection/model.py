# %%
import torch as t
import torch.nn as nn
from torch import Tensor
from utils import multibox_prior, multibox_target, cls_eval, bbox_eval
from datasets import load_data_bananas


def cls_predictor(num_inputs: int, num_anchors: int, num_classes: int):
    """
    锚框有num_anchors+1个类别，因为一个0类是背景。对于每个像素生成num_anchors个锚框,
    一共要对 hwa 个锚框进行分类。

    所以我们的输出为 num_anchors * (num_classes+1) 
    """
    module = nn.Conv2d(num_inputs, num_anchors *
                       (num_classes+1), kernel_size=3, padding=1)
    return module


def bbox_predictor(num_inputs: int, num_anchors: int):
    """
    与类别预测层类似，在这里不同的是，我们需要预测四个坐标，所以是num_anchors*4
    """
    module = nn.Conv2d(num_inputs, num_anchors*4, kernel_size=3, padding=1,)
    return module


def flatten_pred(pred: Tensor):
    """
    由于生成的特征图维度大小不一致，我们需要将张量进行一些处理，变成更一致的格式。
    """
    # permute [batch_size,num_classes*(num_anchors+1) ,h, w]
    # -> [batch_size,h,w,num_classes*(num_anchors+1)]
    # 然后展平-> [batch_size,h*w*num_classes*(num_anchors+1)]
    return t.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds: list[Tensor]):
    """
    将所有预测结果从batch维度进行cat,即保持batch不变
    """
    return t.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels: int, out_channels: int):
    """
    为了在多个尺度下面检测目标，需要将特征图的宽高减半，沿用VGG的设计
    """
    blk = list()

    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                   kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    # 在这里进行宽高减半的操作
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    """
    基本网络块用于从图像中提取特征，我们将特征维度增加，长宽减少为原来的八分之一
    """
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters)-1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


def get_blk(i):
    """
    完整的SSD模型由5个模块组成，每个块生成的特征图既用来生成锚框，又用来预测这些锚框的类别和偏移。
    在这些模块中，第一个是基本网络块，第二到四是高宽减半块，第五个是GMP块，第二到五块是所谓的多尺度特征块
    """
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveAvgPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X: Tensor,
                blk: nn.Module,
                size: list[list[float]],
                ratio: list[list[float]],
                cls_predictor: nn.Conv2d,
                bbox_predictor: nn.Conv2d) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    每个块的前向传播函数，我们不仅仅要最终的结果Y，而且还要分类结果和预测的锚框坐标。
    """
    Y = blk.forward(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor.forward(Y)
    bbox_preds = bbox_predictor.forward(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class TinySSD(nn.Module):
    def __init__(self,
                 num_anchors: int,
                 num_classes: int,
                 sizes: list[list[float]],
                 ratios: list[list[float]],
                 **kwargs) -> None:
        """
        定义SSD的基本架构
        它有5个块组成，其中每个块包含卷积块、分类器和边界框预测器
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.sizes, self.ratios = sizes, ratios
        idx_to_in_channels = [64, 128, 128, 128, 128]

        for i in range(5):
            setattr(self, f"blk_{i}", get_blk(i))
            setattr(self, f"cls_{i}", cls_predictor(
                idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f"bbox_{i}", bbox_predictor(
                idx_to_in_channels[i], num_anchors))

    def forward(self, X: Tensor):
        anchors, cls_preds, bbox_preds = [None]*5, [None]*5, [None]*5,

        # 逐级进行forward并保存中间结果
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X,
                getattr(self, f"blk_{i}"),
                self.sizes[i],
                self.ratios[i],
                getattr(self, f"cls_{i}"),
                getattr(self, f"bbox_{i}")
            )
        anchors = t.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes+1)

        bbox_preds = concat_preds(bbox_preds)

        return anchors, cls_preds, bbox_preds


def cal_loss(cls_preds: Tensor, cls_labels: Tensor, bbox_preds: Tensor, bbox_labels: Tensor, bbox_masks: Tensor):
    """
    计算损失，包括类别的交叉熵和边界框的MAE
    """
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = nn.CrossEntropyLoss(reduction="none").forward(cls_preds.reshape(-1, num_classes),
                                                        cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)

    bbox = nn.L1Loss(reduction="none").forward(
        bbox_preds*bbox_masks, bbox_labels*bbox_masks).mean(dim=1)
    return cls+bbox


class SigleShotMultiboxDetection():
    def __init__(self, data_root, num_epochs, batch_size, cuda, lr, weight_decay, num_classes, num_anchors, sizes, ratios) -> None:
        self.batch_size = batch_size
        self.device = t.device("cuda:0" if cuda else "cpu")
        self.lr, self.weight_decay = lr, weight_decay
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.model = TinySSD(
            num_anchors=num_anchors, num_classes=num_classes, sizes=sizes, ratios=ratios
        ).to(self.device)
        self.train_iter, self.test_iter = load_data_bananas(
            data_root, self.batch_size)
        self.optimizer = t.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            for features, target in self.train_iter:
                self.optimizer.zero_grad()
                X, Y = features.to(self.device), target.to(self.device)
                # 生成多尺度的锚框
                anchors, cls_preds, bbox_preds = self.model.forward(X)
                # 标注偏移量和类别
                bbox_labels, bbox_masks, cls_labels = multibox_target(
                    anchors, Y)
                # 根据类别和偏移量计算损失
                l = cal_loss(cls_preds, cls_labels, bbox_preds,
                             bbox_labels, bbox_masks)
                l.mean().backward()
                self.optimizer.step()

            print(f"ep[{epoch}/{self.num_epochs}], loss:{l.item()},bbox_MAE:{bbox_eval(bbox_preds,bbox_labels,bbox_masks)},cls_ACC:{cls_eval(cls_preds,cls_labels)}")


# %% TEST
if __name__ == "__main__":
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
             [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1
    net = TinySSD(num_anchors, num_classes=1, sizes=sizes, ratios=ratios)
    X = t.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)

    print('output anchors:', anchors.shape)
    print('output class preds:', cls_preds.shape)
    print('output bbox preds:', bbox_preds.shape)
