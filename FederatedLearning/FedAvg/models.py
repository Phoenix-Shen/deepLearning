import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import numpy as np
from utils import load_mnist
import copy
import torch.nn.functional as F
from tqdm import tqdm


class CNNMnist(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(CNNMnist, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net.forward(x)


class CNNCifar(nn.Module):
    def __init__(self, in_channels, num_classes):
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )


class FedAvg(object):
    def __init__(self, args: dict):
        # globale_net
        self.global_net = CNNMnist(args["in_channels"], args["num_classes"])
        # dataloaders for each usr
        self.data_loaders, self.testloader = load_mnist(
            args["iid"], args["num_users"], args["batch_size"])
        # local models
        self.local_nets = [LocalModel(
            idx=i,
            args=args,
            net=CNNMnist(args["in_channels"], args["num_classes"]),
            dataLoader=self.data_loaders[i]
        ) for i in range(args["num_users"])]
        # save configuration
        self.args = args
        self.device = t.device("cuda" if t.cuda.is_available(
        ) and args["device"] == "cuda" else "cpu")
        self.global_net.to(self.device)
        # summary writer
        self.writer = SummaryWriter(log_dir=os.path.join(
            args["log_dir"], "global_test_loss"))
        # add AOI(age of information)
        self.aoi = np.zeros((args["num_users"]), dtype=np.int64)

    def send_parameters(self):
        """
        send the global parameters to all local models.
        """
        [self.local_nets[i].net.load_state_dict(
            self.global_net.state_dict()) for i in range(self.args["num_users"])]

    def aggregate(self, usr_idxs: list[int]):
        """
        aggregate all local models.
        """
        weight_locals = [self.local_nets[usr_idx].net.state_dict()
                         for usr_idx in usr_idxs]
        weight_avg = copy.deepcopy(weight_locals[0])
        # perform fed avg algorthm
        for key in weight_locals[0].keys():
            for i in range(1, len(weight_locals)):
                weight_avg[key] = weight_avg[key] + weight_locals[i][key]
            weight_avg[key] = t.div(weight_avg[key], len(weight_locals))
        self.global_net.load_state_dict(weight_avg)

    def eval(self, dataloader: DataLoader):
        """
        evaluate the global model on the specified dataset.
        """
        self.global_net.eval()
        correct = 0
        test_loss = 0
        with t.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                log_probs = self.global_net.forward(images)
                # sum up the batch loss
                test_loss += F.cross_entropy(log_probs,
                                             labels, reduction="sum").item()
                # equals log_probs.argmax(1, True)
                y_prediction = log_probs.data.max(1, keepdim=True)[1]
                correct += y_prediction.eq(labels.data.view_as(y_prediction)
                                           ).long().cpu().sum()
        test_loss /= len(dataloader.dataset)
        acc = 100.*correct / len(dataloader.dataset)
        return acc, test_loss

    def train(self):
        """
        start the training procedre
        """
        for ep in tqdm(range(self.args["epoch"])):
            # switch to train mode
            self.global_net.train()
            # choose users for training
            idx_users = self.choose_users()
            for idx_usr in idx_users:
                self.local_nets[idx_usr].train()
            # perform aggrate opeartion
            self.aggregate(idx_users)
            # send parameters to all local models
            self.send_parameters()
            # switch to evaluation mode
            acc, loss = self.eval(self.testloader)
            print("\rep:{},acc_test:{},loss_test:{}".format(ep, acc, loss), end="")

    def choose_users(self):
        """
        choose users to train the model
        """
        # get the user number for fedavg
        avg_usr_num = int(self.args["fraction"]*self.args["num_users"])
        assert avg_usr_num >= 1
        # 3 strategies
        if self.args["fed_strategy"] == "rand":

            idx_users = np.random.choice(
                range(self.args["num_users"]), avg_usr_num,
                replace=False)
        elif self.args["fed_strategy"] == "aoi-biggest":
            # asending order
            idx_users = np.argsort(self.aoi)[-avg_usr_num:]
        elif self.args["fed_strategy"] == "aoi-smallest":
            # asending order
            idx_users = np.argsort(self.aoi)[:avg_usr_num]

        # update aoi
        self.aoi += 1
        self.aoi[idx_users] = 0
        return idx_users


class LocalModel(object):
    def __init__(self, idx: int, args: dict, net: nn.Module, dataLoader: DataLoader):
        self.idx = idx
        self.net = net
        self.device = t.device("cuda" if t.cuda.is_available(
        ) and args["device"] == "cuda" else "cpu")
        self.optimizer = t.optim.Adam(self.net.parameters(
        ), lr=args["lr"], weight_decay=args["weight_decay"])
        self.dataLoader = dataLoader
        self.args = args
        self.writer = SummaryWriter(log_dir=os.path.join(
            args["log_dir"], "local_model_{}".format(self.idx)))
        self.loss_fn = nn.CrossEntropyLoss()
        self.global_step = 0

    def train(self):
        """
        strat training process
        """
        self.net.train()
        self.net.to(self.device)
        for epoch in range(self.args["local_epochs"]):
            batch_loss = []
            for batch_dix, (img, label) in enumerate(self.dataLoader):
                img = img.to(self.device)
                label = label.to(self.device)
                output = self.net.forward(img)
                self.optimizer.zero_grad()
                loss = self.loss_fn(output, label)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())

            self.writer.add_scalar("loss_local", np.mean(
                batch_loss), self.global_step)
            self.global_step += 1
