"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time

import torch
import pickle

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
# from core.test_engine import run_inference
import utils.logging
from modeling import model_builder
import utils.net as net_utils
from utils.detectron_weight_helper import load_detectron_weight
import nn as mynn
import utils.blob as blob_utils
import numpy as np

from datasets.crack_dataset import CrackDataSet, collate_minibatch
from datasets.cifa_dataset import cifar10_dataset
from modeling.basenet import BaseNet
from grad_cam import ShowGradCam
from torchvision import transforms
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = BaseNet(cfg.MODEL.NUM_CLASSES)
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(
            load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    # model = mynn.DataParallel(
    #     model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    # logger.info('Called with args:')
    # logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test',
            os.path.basename(ckpt_path).split('.')[0])
        logger.info('Automatically set output directory to %s',
                    args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis
    cfg.MODEL.NUM_CLASSES = 2

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    assert_and_infer_cfg()

    # logger.info('Testing with config:')
    # logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    if args.load_ckpt:
        while not os.path.exists(args.load_ckpt):
            logger.info('Waiting for {} to exist...'.format(args.load_ckpt))
            time.sleep(10)
    if args.load_detectron:
        while not os.path.exists(args.load_detectron):
            logger.info('Waiting for {} to exist...'.format(
                args.load_detectron))
            time.sleep(10)

    _, dataloader = cifar10_dataset()
    dataiterator = iter(dataloader)

    model = BaseNet()
    logger.info("loading checkpoint %s", args.load_ckpt)
    checkpoint = torch.load(args.load_ckpt)
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()

    correct = 0
    noncorrect = 0
    cnt = 0

    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    labels = pickle.load(open('/home/syb/documents/Crack_Image_WSOD/data/cifar10_images/labels.pkl', 'rb'))

    for _ in range(500):
        if cnt >= 10:
            exit(0)
        # try:
        #     input_data = next(dataiterator)
        # except StopIteration:
        #     dataiterator = iter(dataloader)
        #     input_data = next(dataiterator)
        image_name = '/home/syb/documents/Crack_Image_WSOD/data/cifar10_images/img' + str(cnt) + '.png'
        image = cv2.imread(image_name)
        input_data = transform(image).unsqueeze(0)
        label = torch.Tensor([labels[cnt]])
        # print(input_data)
        # print(label)

        gradCam = ShowGradCam(model.base_network.layer4[2].conv3)
        # print(input_data[0])

        cnt += 1
        imgs_clf, _ = model(input_data.to('cuda'), label.to('cuda'))

        scores = imgs_clf.data.cpu().numpy()

        clf = np.argmax(scores)

        label = label.numpy().squeeze(axis=0)
        # print(clf, label)
        model.zero_grad()
        
        imgs_clf[0, clf].backward()

        gradCam.show_on_img(image, str(cnt))
        cv2.imwrite('output_pic/' + str(cnt) + '_origin.jpg', image)

        if clf == label:
            correct += 1
        else:
            noncorrect += 1

    print("Accuracy : %3d/%3d, %.1f%%" % (correct,
          correct + noncorrect, correct * 100 / (correct + noncorrect)))
