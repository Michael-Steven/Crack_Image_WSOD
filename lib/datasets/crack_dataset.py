from typing import Tuple
import torch.utils.data as data
import numpy as np
from core.config import cfg
from torch.utils.data.dataloader import default_collate
import cv2
import torch

def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data', 'labels']
    return blob_names


def get_max_shape(im_shapes):
    """Calculate max spatial size (h, w) for batching given a list of image shapes
    """
    max_shape = np.array(im_shapes).max(axis=0)
    assert max_shape.size == 2
    # Pad the image so they can be divisible by a stride
    return max_shape


def im_list_to_blob(ims):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
    Output is a 4D HCHW tensor of the images concatenated along axis 0 with
    shape.
    """
    if not isinstance(ims, list):
        ims = [ims]
    max_shape = get_max_shape([im.shape[:2] for im in ims])

    num_images = len(ims)
    blob = np.zeros(
        (num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    image_path = cfg.IMAGEPATH
    blob = dict()
    for i in range(num_images):
        im = cv2.imread(image_path + roidb[i]['image']).astype(np.float32, copy=False)
        im = cv2.resize(im, (400, 400))
        # im -= cfg.PIXEL_MEANS
        blob['image'] = image_path + roidb[i]['image']
        im /= 255.0
        im -= np.array([[[0.5068035, 0.5068035, 0.5068035]]])
        im /= np.array([[[0.16240637, 0.16240637, 0.16240637]]])
        processed_ims.append(im)
    blob['data'] = im_list_to_blob(processed_ims)
    return blob


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob = _get_image_blob(roidb)
    blobs['data'] = im_blob['data']
    blobs['image'] = im_blob['image']

    rois_blob = np.zeros((0, 4), dtype=np.float32)
    labels_blob = np.zeros((0, 1), dtype=np.float32)

    num_images = len(roidb)
    for im_i in range(num_images):
        rois = roidb[im_i]['bbox']
        labels = np.zeros((1, 1), dtype=np.float32)
        labels[0][0] = roidb[im_i]['label']
        for roi in rois:
            # bboxs: bach_idx, x1, y1, x2, y2
            rois_blob_this_image = np.array(
                [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]])
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        labels_blob = np.vstack((labels_blob, labels))

    blobs['rois'] = rois_blob
    blobs['labels'] = labels_blob

    return blobs


class CrackDataSet(data.Dataset):
    def __init__(self, proposal_files, num_classes, training=True):
        self.proposal_files = proposal_files
        self._num_classes = num_classes
        self.training = training
        self.DATA_SIZE = len(self.proposal_files)
        # print(type(images[0]))
        # print(type(proposal_files[0]))

    def __getitem__(self, index):
        single_db = [self.proposal_files[index]]
        blobs = get_minibatch(single_db, self._num_classes)
        # print(image.shape)
        # print(len(proposal['bbox']))
        blobs['data'] = blobs['data'].squeeze(axis=0)
        # blobs['rois'] = blobs['rois']
        blobs['labels'] = blobs['labels'].squeeze(axis=0).squeeze(axis=0)
        # print(blobs)
        return blobs

    def __len__(self):
        return self.DATA_SIZE


def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    lists = []
    for i in range(0, len(list_of_blobs)):
        new_blob = dict({'data': list_of_blobs[i].pop('data'),
                      'rois': list_of_blobs[i].pop('rois'),
                      'image': list_of_blobs[i].pop('image'),
                      'labels': list_of_blobs[i].pop('labels')})
        batch_ind = i * np.ones((new_blob['rois'].shape[0], 1))
        new_blob['rois'] = np.hstack((batch_ind, new_blob['rois']))
        lists.append(new_blob)
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = lists[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        tmp_list = [item.copy() for item in mini_list]
        for list_item in tmp_list:
            del list_item['rois']
        minibatch = default_collate(tmp_list)

        # concat all rois in the catch
        cat_result = torch.from_numpy(mini_list[0]['rois'])
        for list_item in mini_list[1:]:
            cat_result = torch.cat((cat_result, torch.from_numpy(list_item['rois'])), 0)
        minibatch['rois'] = cat_result.to(torch.float32)
        for key in minibatch:
            Batch[key] = minibatch[key]
    return Batch
