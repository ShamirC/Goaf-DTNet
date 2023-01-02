import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision
import os

# using tensorboard to record and visualizing data
# loss, precision, recall, f1, mIoU,
# interim feature maps
class CrackTensorBoard():
    def __init__(self, cfg):
        super(CrackTensorBoard, self).__init__()

        # tensorboard log file
        logs = os.path.join(cfg.SAVE.LOG_EXP,cfg.SAVE.LOG_NAME)
        self.writer = SummaryWriter(logs)

    def batch_loss(self, phase,step,loss):
        # loss curve of all iterations
        # loss: Dict, {"side1_segloss": xxx} or {"side1_extloss":xxx}
        for key, value in loss.items():
            self.writer.add_scalar("Loss" + "/" + phase + "-" + key, value, step)

    def epoch_loss(self,phase,step,loss):
        # loss curve of all epochs
        # loss: Dict, {"sidel1_segloss":xxx} or {"side1_extloss":xxx}
        for key, value in loss.items():
            self.writer.add_scalar("Loss" + "/" + phase + "-" + key, value, step)

    def criteria(self,phase,step,accuracy):
        # metircs
        for key, value in accuracy.items():
            self.writer.add_scalar("Accuracy" + "/" + phase + "-" + key, value, step)

    def fmap(self,phase,step,images,dataformats="NCHW"):
        # images: {'image': input image(tensor), 'label': ture label(tensor), 'side1_segpred': xx, ... , "fused_extpred": xxx,

       for key, value in images.items():
           self.writer.add_images(phase + "/" + key, value, step, dataformats=dataformats)


class Record():
    def __init__(self):
        self.epoch_value =[]

        self.count = 0

    def update(self,value):
        self.epoch_value.append(value)
        self.count += 1

    def average(self):
        __avg = sum(self.epoch_value) / self.count
        return __avg


class Metrics():
    def __init__(self, preds, labels):
        """
        :param preds: Tensor, [n,c,h,w]
        :param labels: Tensor, [n,c,h,w]
        """
        self.preds = F.sigmoid(preds).clone()
        self.preds = self.preds.squeeze(1).cpu().detach().numpy()
        self.preds = (self.preds>0.5).astype(np.uint8)

        self.labels = labels.clone()
        self.labels = self.labels.cpu().detach().numpy().astype(np.uint8)

        self.eps = 1e-6

    def f1_score(self):
        tp,fp,fn = self.get_statistics()

        f = (tp) / ((2*tp + fn + fp)+self.eps)

        return f

    def iou(self):
        tp,fp,fn = self.get_statistics()
        iou = tp / ((tp + fp + fn)+self.eps)
        return iou

    def get_statistics(self):
        tp = np.sum((self.preds == 1) & (self.labels == 1))
        fp = np.sum((self.preds == 1) & (self.labels == 0))
        fn = np.sum((self.preds == 0) & (self.labels == 1))

        return [tp,fp,fn]



