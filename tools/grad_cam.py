# coding: utf-8
import cv2
import numpy as np
import torch

# 类的作用
# 1.编写梯度获取hook
# 2.网络层上注册hook
# 3.运行网络forward backward
# 4.根据梯度和特征输出热力图


class ShowGradCam:
    def __init__(self, conv_layer):
        assert isinstance(
            conv_layer, torch.nn.Module), "input layer should be torch.nn.Module"
        self.conv_layer = conv_layer
        self.conv_layer.register_forward_hook(self.farward_hook)
        self.conv_layer.register_backward_hook(self.backward_hook)
        self.grad_res = []
        self.feature_res = []

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_res.append(grad_out[0].detach())

    def farward_hook(self, module, input, output):
        self.feature_res.append(output)

    def gen_cam(self, feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:],
                       dtype=np.float32)  # cam shape (H, W)
        weights = np.mean(grads, axis=(1, 2))  #

        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]

        # relu
        cam = np.maximum(cam, 0)
        # cam = cv2.resize(cam, (400, 400))
        cam -= np.min(cam)
        cam /= np.max(cam)
        # cam_min, cam_max = np.min(cam), np.max(cam)
        # cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def show_on_img(self, input_img, cnt):
        '''
        write heatmap on target img
        :param input_img: cv2:ndarray/img_pth
        :return: save jpg
        '''
        if isinstance(input_img, str):
            input_img = cv2.imread(input_img)
        img_size = (input_img.shape[1], input_img.shape[0])
        # print(img_size)
        fmap = self.feature_res[0].cpu().data.numpy().squeeze()
        grads_val = self.grad_res[0].cpu().data.numpy().squeeze()
        cam = self.gen_cam(fmap, grads_val)
        cam = cv2.resize(cam, img_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)#/255.
        # cam = heatmap + np.float32(input_img/255.)
        # cam = cam / np.max(cam)*255
        cam = 0.3 * heatmap + 0.7 * input_img
        cv2.imwrite('output_pic/' + cnt + '_grad_feature.jpg', cam)
        # print('save gradcam result in grad_feature.jpg')
