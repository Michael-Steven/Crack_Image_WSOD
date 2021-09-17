import torch.utils.data as data
import pickle

class CrackDataSet(data.Dataset):
    def __init__(self, images, proposal_files, training=True):
        self.images = images
        self.proposal_files = proposal_files
        self.training = training
        self.DATA_SIZE = len(self.images)
        # print(type(images[0]))
        # print(type(proposal_files[0]))

    def __getitem__(self, index):
        image = self.images[index]
        proposal = self.proposal_files[index]
        # print(image.shape)
        # print(len(proposal['bbox']))
        return image, proposal

    def __len__(self):
        return self.DATA_SIZE