import torch
import torch.nn as nn
import modeling.resnet as resnet
import torch.nn.functional as F
from mmcv.ops import RoIPool
from core.config import cfg


class BaseNet(nn.Module):
    # def __init__(self, num_class, model_path, pretrain_choice='ImageNet', **kwargs):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.num_class = 10
        self.base_network = resnet.ResNet()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.base_network.dim_out, self.num_class)

        # hidden_dim = 1024
        # roi_size = 7
        # self.roi_pooling = RoIPool(output_size=roi_size, spatial_scale=1. / 8.)
        # self.fc1 = nn.Linear(self.base_network.dim_out * roi_size**2, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.mil_score0 = nn.Linear(hidden_dim, 2)
        # self.mil_score1 = nn.Linear(hidden_dim, 2)
        self.criterion = torch.nn.CrossEntropyLoss()

        # self._init_weights()


    def mil_losses(self, cls_score, labels):
        loss = torch.tensor([]).to('cuda')
        for i in range(len(cls_score)):
            cls_score_tmp = cls_score[i].clamp(1e-6, 1 - 1e-6)
            labels_tmp = labels[i].clamp(0, 1)
            batch_item_loss = -labels_tmp * torch.log(cls_score_tmp) - (1 - labels_tmp) * torch.log(1 - cls_score_tmp)
            loss = torch.cat((loss, batch_item_loss), 0)
        # loss = self.criterion(cls_score, labels.to(torch.long))
        return loss.mean()

    # def _init_weights(self):
    #     nn.init.normal_(self.mil_score0.weight, std=0.01)
    #     nn.init.constant_(self.mil_score0.bias, 0)
    #     nn.init.normal_(self.mil_score1.weight, std=0.01)
    #     nn.init.constant_(self.mil_score1.bias, 0)

    def forward(self, data, labels):
        # print(data)
        # print(labels)
        # exit(0)
        # backbone
        x = self.base_network(data)

        features = self.gap(x)
        features = features.view(features.size(0), -1)
        # print(imgs.size(), features.size())
        
        # classification
        score = self.classifier(features)
        # print(score.shape)
        # exit(0)
        # print(score)
        cls_loss = self.criterion(score, labels.long())
        # print(clf_loss)
        if self.training:
            return cls_loss
        else:
            return score, cls_loss

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.5 * initial_lr},
            {'params': self.classifier.parameters(), 'lr': 1.0 * initial_lr},
        ]
        return params

    def load_param(self, pre_trained_path):
        # for key in self.state_dict():
            # print(self.state_dict()[key])
            # print(key)
        
        param_dict = torch.load(pre_trained_path)
        for i in param_dict:
            # if 'classifier' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict[i])
