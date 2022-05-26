
# %%
import torch as t
import os
import torchvision as tv
import pandas as pd


def read_data_bananas(data_root, is_train=True):
    """
    读取香蕉检测数据集
    ------
    Params:
        data_root: the datasaet root
        is_train: load the train set or the test set
    Returns:
        All images and its targets
    """
    csv_fname = os.path.join(
        data_root, "bananas_train" if is_train else "bananas_val", "label.csv")

    # 读CSV
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index("img_name")
    # 准备提取数据
    images, targets = [], []

    for img_name, target in csv_data.iterrows():
        images.append(
            tv.io.read_image(
                os.path.join(
                    data_root, "bananas_train" if is_train else "bananas_val", "images", f"{img_name}")
            )
        )
        # raget 包含了 类别，左上x，y，右下x，y
        # 其中所有的图像都具有相同的香蕉类（0类）
        targets.append(list(target))
    # 这里除以256是将坐标归一化
    return images, t.tensor(targets).unsqueeze(1)/256


class BananasDataset(t.utils.data.Dataset):
    """
    香蕉检测数据集
    """

    def __init__(self, data_root, is_train=True) -> None:
        super().__init__()
        self.features, self.labels = read_data_bananas(
            data_root=data_root, is_train=is_train)
        print("read " + str(len(self.features)) +
              (" training examples" if is_train else " validation examples"))

    def __getitem__(self, idx: int):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


def load_data_bananas(data_root, batch_size):
    """
    加载香蕉检测数据集
    """
    train_iter = t.utils.data.DataLoader(
        BananasDataset(data_root, True), batch_size=batch_size, shuffle=True
    )

    val_iter = t.utils.data.DataLoader(
        BananasDataset(data_root, False), batch_size=batch_size, shuffle=False
    )

    return train_iter, val_iter


# %% TEST
if __name__ == "__main__":
    batch_size = 32
    train_iter, _ = load_data_bananas(
        r"../../dataset\banana-detection", batch_size)
    for img, label in train_iter:
        print(img.shape, label.shape)
        break

# %%
