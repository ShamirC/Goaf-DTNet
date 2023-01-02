# To build the goaf crack segmentation datasets, distigunised from goaf crack line extraction datasets
from torch.utils import data
import cv2
import numpy as np


class GoafCrackSeg(data.Dataset):
    def __init__(self,cfg, is_train,transforms=None):
        super(GoafCrackSeg, self).__init__()
        # print(is_train)
        self.transforms = transforms
        if is_train:
            data_list = cfg.DATA.DATA_LIST_TRAIN
        else:
            data_list = cfg.DATA.DATA_LIST_VAL

        infos = [line.split() for line in open(data_list).readlines()] # all the lines in txt file, line.split() remove the blank in the start and end.
        self.image_paths = [info[0] for info in infos]
        self.label_seg_paths = [info[1] for info in infos]
        self.label_line_paths = [info[2] for info in infos]

    def __getitem__(self, item):
        image =cv2.cvtColor(cv2.imread(self.image_paths[item]),cv2.COLOR_BGR2RGB)   # Shape of (H, W, C)
        label_seg = cv2.imread(self.label_seg_paths[item],cv2.IMREAD_GRAYSCALE)   # Shape of (H, W)
        label_line = cv2.imread(self.label_line_paths[item],cv2.IMREAD_GRAYSCALE)  # Shape of (H, W)

        if self.transforms is not None:
            image, label_seg, label_line = self.transforms(image,label_seg, label_line)

        image, label_seg, label_line = self.preprocess(image, label_seg, label_line)

        return image, label_seg, label_line

    def __len__(self):
        return len(self.image_paths)

    def preprocess(self,image,label_seg, label_line):
        # coverting image shape from (H,W,C) into (C,H,W)
        image = image.transpose((2,0,1)).astype(np.float32)

        return image, label_seg/255.0, label_line/255.0

