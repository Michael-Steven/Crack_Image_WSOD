import torch
import torch.nn as nn
import modeling.resnet as resnet
import torch.nn.functional as F
from torchvision.ops import RoIPool
from core.config import cfg

class BaseNet(nn.Module):
    # def __init__(self, num_class, model_path, pretrain_choice='ImageNet', **kwargs):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.num_class = cfg.MODEL.NUM_CLASSES
        self.base_network = resnet.ResNet()

        # self.in_planes = 2048
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(self.in_planes, num_class)


        hidden_dim = 2048
        roi_size = 7
        self.fc1 = nn.Linear(self.base_network.dim_out * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mil_score0 = nn.Linear(self.base_network.dim_out, 2)
        self.mil_score1 = nn.Linear(self.base_network.dim_out, 2)
        self.criterion = torch.nn.CrossEntropyLoss()

        self._init_weights()


        # if model_path == '':
        #     model_path = '/home/youmeiyue/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'
        # if pretrain_choice == 'ImageNet':
        #     self.base_network.load_param(model_path)
        #     print('Loading pretrained ImageNet model')
        # elif pretrain_choice == 'BaselineTest':
        #     self.load_param(model_path)
        #     print('Loading pretrained BaselineTest model')


    def _init_weights(self):
        nn.init.normal_(self.mil_score0.weight, std=0.01)
        nn.init.constant_(self.mil_score0.bias, 0)
        nn.init.normal_(self.mil_score1.weight, std=0.01)
        nn.init.constant_(self.mil_score1.bias, 0)


    def forward(self, data, rois, labels):
        # backbone
        x = self.base_network(data)

        # box_feat
        x = RoIPool(outputsize=7, spatial_scale=1. / 8.)(x, rois)
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # mil
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        mil_score0 = self.mil_score0(x)
        mil_score1 = self.mil_score1(x)
        mil_score = F.softmax(mil_score0, dim=0) * F.softmax(mil_score1, dim=1)
        im_cls_score = mil_score.sum(dim=0)
        cls_loss = self.criterion(im_cls_score, labels.long())

        # features = self.gap(features)
        # features = features.view(features.size(0), -1)
        # # print(imgs.size(), features.size())
        
        # # classification
        # imgs_clf = self.classifier(features)
        # clf_loss = self.criterion(imgs_clf, labels.long())
        # # print(clf_loss)
        if self.training:
            return cls_loss
        else:
            return mil_score, cls_loss

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

