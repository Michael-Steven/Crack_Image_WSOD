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
from modeling.basenet import BaseNet

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
    logger.info('Called with args:')
    logger.info(args)

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

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

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

    test_proposal_files = pickle.load(open(cfg.TEST_PROPOSAL_FILE_PATH, 'rb'))
    dataset = CrackDataSet(
        test_proposal_files,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    # num_epoch = 1     # number of epochs to train on
    batch_size = 1  # training batch size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
        # collate_fn=collate_minibatch)
    dataiterator = iter(dataloader)
    test_size = len(dataset)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    model = BaseNet(cfg.MODEL.NUM_CLASSES)
    logger.info("loading checkpoint %s", args.load_ckpt)
    checkpoint = torch.load(args.load_ckpt)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for _ in range(test_size):
        try:
            input_data = next(dataiterator)
        except StopIteration:
            dataiterator = iter(dataloader)
            input_data = next(dataiterator)

        # print(input_data)
        imgs_clf, _ = model(input_data['data'], input_data['labels'])

        # net_outputs = model(**input_data)
        scores = imgs_clf.data.numpy()
        labels = input_data['labels']

        score = scores.sum(axis=0)
        label = labels.numpy().squeeze(axis=0)

        print(score, label)
        
        if label == 1.0 and score[1] > score[0]:
            tp += 1
        if label == 0.0 and score[1] > score[0]:
            fp += 1
        if label == 0.0 and score[0] > score[1]:
            tn += 1
        if label == 1.0 and score[0] > score[1]:
            fn += 1

    print("TP: %3d, FP: %3d, TN: %3d, FN: %3d\nAccuracy : %3d/%3d, %.1f%%\nRecall   : %3d/%3d, %.1f%%\nPrecision: %3d/%3d, %.1f%%\n" %
          (tp, fp, tn, fn, tp + tn, tp + fp + tn + fn, (tp + tn) / (tp + fp + tn + fn) * 100,
           tp, tp + fn, tp / (tp + fn) * 100, tp, tp + fp, tp / (tp + fp) * 100))
