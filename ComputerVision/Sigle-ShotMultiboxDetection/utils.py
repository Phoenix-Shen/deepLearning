# %%
from torch import Tensor
import torch as t


def multibox_prior(data: Tensor, sizes: list[list[float]], ratios: list[list[float]]):
    """
    生成以每个像素为中心具有不同形状的锚框,请注意下面都是将宽高当做1来算的，可能这么解释有些不清楚，但是你需要详细读一下代码
    就知道我在说什么了。O(∩_∩)O
    """
    # data.shape = [batch,channels,height,width]
    # -2 就是数组的倒数第二个坐标
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # 对于每个像素需要生成(num_sizes+num_ratios-1)这么多个锚框
    boxes_per_pixel = (num_sizes+num_ratios-1)
    # 转tensor
    size_tensor = t.tensor(sizes).to(device)
    ratio_tensor = t.tensor(ratios).to(device)

    # 为了移动将锚点移动到像素中心，需要设置偏移量。
    # 因为一个像素的高和宽为1，所以设置xy的offset均为0.5
    offset_h, offset_w = 0.5, 0.5
    # 获得缩放的步长
    step_h = 1.0/in_height
    step_w = 1.0/in_width
    # 生成所有锚框的中心点,先arange出来然后再乘以缩放的步长
    center_h = (t.arange(in_height, device=device)+offset_h)*step_h
    center_w = (t.arange(in_width, device=device)+offset_w)*step_w
    # 使用t.meshgrid API来生成(x,y)坐标
    shift_y, shift_x = t.meshgrid(center_h, center_w, indexing="ij")
    # 展平
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 然后生成"boxes_per_pixel"个高和宽,*in_height/in_width是为了做归一化，生成正方形的锚框
    # 详情看下面的评论https://zh-v2.d2l.ai/chapter_computer-vision/anchor.html
    w = t.cat(
        (size_tensor*t.sqrt(ratio_tensor[0]), sizes[0]
         * t.sqrt(ratio_tensor[1:]))
    )*in_height/in_width

    h = t.cat(
        (size_tensor/t.sqrt(ratio_tensor[0]), sizes[0]
         / t.sqrt(ratio_tensor[1:]))
    )
    # 除以2来获得半高和半宽
    anchor_manipulations = t.stack((-w, -h, w, h))
    anchor_manipulations = anchor_manipulations.T
    # https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html?highlight=repeat_interleave#torch.repeat_interleave
    # t.repeat : This is different from torch.Tensor.repeat() but similar to numpy.repeat.
    anchor_manipulations = anchor_manipulations.repeat(in_height*in_width, 1)/2
    # 每个中心点，都有boxes_per_pixel个锚框
    # 所以要repeat_interleave，重复boxes_per_pixel次
    out_grid = t.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
    out_grid = out_grid.repeat_interleave(boxes_per_pixel, dim=0)

    # 再加上偏移就得到了所有锚框的四个坐标
    output = out_grid+anchor_manipulations
    return output.unsqueeze(0)


def box_iou(boxes1, boxes2):
    """
    计算两个锚框或者边界框列表中成对的交并比，一共有num_box1*num_box2个组合
    """
    # 定义如何计算
    def box_area(boxes): return (
        (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]))
    # 计算每个Box的面积
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # 将它们变成 [num_box1,num_box2]的形状，直接并行计算
    # None的作用就是unsqueeze，并不是什么高大上的东西
    inter_upperlefts = t.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = t.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)

    # 计算交并面积
    inter_areas = inters[:, :, 0]*inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas/union_areas


def assign_anchor_to_bbox(ground_truth: Tensor, anchors: Tensor, device: str, iou_threshold=.5):
    """
    将最接近的真实边界框分配给锚框
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 计算IOU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配真实边界框的张量
    # Creates a tensor of size size filled with fill_value. The tensor’s dtype is inferred from fill_value.
    anchors_bbox_map = t.full((num_anchors,), -1, dtype=t.long, device=device)
    # 根据阈值决定是否分配真实边界框
    max_ious, indices = t.max(jaccard, dim=1)
    anc_i = t.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious > iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    # 先定义填充变量
    col_discard = t.full((num_anchors,), -1)
    row_discard = t.full((num_gt_boxes,), -1)
    # 然后一个个分配
    for _ in range(num_gt_boxes):
        max_idx = t.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx/num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def box_corner_to_center(boxes: Tensor):
    """
    从 (左上,右下)转换到(center,w,h)
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # center
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    w = x2-x1
    h = y2-y1
    boxes = t.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes: Tensor):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = t.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def offset_bboxes(anchors: Tensor, assigned_bb: Tensor, eps=1e-6):
    """
    对锚框偏移量进行转换
    """
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)

    offset_xy = 10*(c_assigned_bb[:, :2]-c_anc[:, :2])/c_anc[:, 2:]
    offset_wh = 5*t.log(eps+c_assigned_bb[:, 2:]/c_anc[:, 2:])
    offset = t.cat([offset_xy, offset_wh], axis=1)
    return offset


def multibox_target(anchors: Tensor, labels: Tensor):
    """
    使用真实边界框标记锚框的类别和偏移量，在模型中用这个做回归
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        # print(anchors_bbox_map.shape)
        bbox_mask = (
            (anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # 将类标签和分配的边界框初始化为0
        class_labels = t.zeros(num_anchors, dtype=t.long, device=device)
        assigned_bb = t.zeros((num_anchors, 4), dtype=t.float32, device=device)
        # 使用真实边界框来标记锚框的类别，如果锚框没有分配则为背景
        indices_true = t.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long()+1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_bboxes(anchors, assigned_bb)*bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = t.stack(batch_offset)
    bbox_mask = t.stack(batch_mask)
    class_labels = t.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def cls_eval(cls_preds: Tensor, cls_labels: Tensor):
    """
    预测类别的准确度
    """
    results = (cls_preds.argmax(
        dim=-1).type_as(cls_labels.dtype) == cls_labels).sum()
    return float(results)


def bbox_eval(bbox_preds: Tensor, bbox_labels: Tensor, bbox_masks: Tensor):
    """
    预测锚框与边界框的偏移
    """
    results = (t.abs((bbox_preds-bbox_labels)*bbox_masks)).sum()
    return float(results)


# %% TEST1
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    try:
        img = plt.imread("../../figures/cat.jpg")
        print("read ../../figures/cat.jpg")
    except:
        img = plt.imread("./figures/cat.jpg")
        print("read figures/cat.jpg")
    h, w = img.shape[:2]
    print(h, w)
    X = t.rand(size=(1, 3, h, w))
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print(Y.shape)

    # Show BBOXES
    def bbox_to_rect(bbox, color):
        # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
        # ((左上x,左上y),宽,高)
        return plt.Rectangle(
            xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
            fill=False, edgecolor=color, linewidth=2)

    def show_bboxes(axes, bboxes: Tensor, labels=None, colors=None):
        """
        在图片上显示锚框
        """
        def _make_list(obj, defalut_values=None):
            if obj is None:
                obj = defalut_values
            elif not isinstance(obj, (list, tuple)):
                obj = [obj]
            return obj

        labels = _make_list(labels)
        colors = _make_list(colors, ["b", "g", "r", "m", "c"])

        for i, bbox in enumerate(bboxes):
            color = colors[i % len(colors)]
            rect = bbox_to_rect(bbox.detach().numpy(), color)
            axes.add_patch(rect)
            if labels and len(labels) > i:
                text_color = 'k' if color == 'w' else 'w'
                axes.text(rect.xy[0], rect.xy[1], labels[i],
                          va='center', ha='center', fontsize=9, color=text_color,
                          bbox=dict(facecolor=color, lw=0))

    boxes = Y.reshape(h, w, 5, 4)
    bbox_scale = t.tensor((w, h, w, h))
    fig = plt.imshow(img)
    show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
                's=0.75, r=0.5'])
# %% TEST2
if __name__ == '__main__':
    ground_truth = t.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = t.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])

    fig = plt.imshow(img)
    show_bboxes(fig.axes, ground_truth[:, 1:]
                * bbox_scale, ['dog', 'cat'], 'k')
    show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
    labels = multibox_target(anchors.unsqueeze(dim=0),
                             ground_truth.unsqueeze(dim=0))
    print(labels[1], labels[2], labels[0])
