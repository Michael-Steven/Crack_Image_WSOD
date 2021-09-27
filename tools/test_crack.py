"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time

import torch

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
    model = model_builder.Generalized_RCNN()
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

    model = mynn.DataParallel(
        model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

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

    model = initialize_model_from_cfg(args, gpu_id=0)

    im = cv2.imread("/home/syb/documents/Crack_Image_WSOD/data/cut/combine/1_1_1_result.jpg")

    ## region
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    numShowRects = 300
    rois_blob = np.zeros((0, 5), dtype=np.float32)

    inputs = dict()
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal till numShowRects
        if i < numShowRects:
            x, y, w, h = rect
            rois_blob_this_image = np.array([0, x, y, x + w, y + h])
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        else:
            break
    inputs['rois'] = [torch.from_numpy(rois_blob).type(torch.float32)]
    inputs['labels'] = [torch.from_numpy(np.array([[0.0, 0.0]])).type(torch.float32)]


    ## image
    img = im.astype(np.float32, copy=False) - cfg.PIXEL_MEANS
    inputs['data'] = [torch.from_numpy(blob_utils.im_list_to_blob(img))]


    # print(inputs)

    ## inference
    return_dict = model(**inputs)

    scores = return_dict['mil_score'].data.cpu().numpy()

    print(scores.sum(axis=0))

    # print(scores)

    for i in range(len(scores)):
        if scores[i][0] > 0.01 or scores[i][1] > 0.01:
            # print(rois_blob[i][1])
            # print(rois_blob[i][2])
            # print(rois_blob[i][3])
            # print(rois_blob[i][4])
            cv2.rectangle(im, (int(rois_blob[i][1]), int(rois_blob[i][2])), 
                (int(rois_blob[i][3]), int(rois_blob[i][4])), (255, 0, 0), 2)

    # cv2.imwrite("./output.jpg", im)


    # run_inference(
    #     args,
    #     ind_range=args.range,
    #     multi_gpu_testing=args.multi_gpu_testing,
    #     check_expected_results=True)