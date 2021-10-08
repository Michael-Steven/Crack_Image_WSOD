import torch
import torch.nn as nn
import modeling.resnet as resnet

class BaseNet(nn.Module):
    # def __init__(self, num_class, model_path, pretrain_choice='ImageNet', **kwargs):
    def __init__(self, num_class):
        super(BaseNet, self).__init__()
        self.num_class = num_class
        self.base_network = resnet.ResNet()

        self.in_planes = 2048
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.in_planes, num_class)
        self.criterion = torch.nn.CrossEntropyLoss()

        # if model_path == '':
        #     model_path = '/home/youmeiyue/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'
        # if pretrain_choice == 'ImageNet':
        #     self.base_network.load_param(model_path)
        #     print('Loading pretrained ImageNet model')
        # elif pretrain_choice == 'BaselineTest':
        #     self.load_param(model_path)
        #     print('Loading pretrained BaselineTest model')

    def forward(self, data, labels):
        # print(data)
        # print(labels)
        # exit(0)
        features = self.base_network(data)
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        # print(imgs.size(), features.size())
        
        # classification
        imgs_clf = self.classifier(features)
        clf_loss = self.criterion(imgs_clf, labels.long())
        # print(clf_loss)
        if self.training:
            return clf_loss
        else:
            return imgs_clf, clf_loss

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

