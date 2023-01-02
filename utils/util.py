import os
import torch
import numpy as np
import random
import torch.nn as nn
import logging
from prettytable import PrettyTable
from collections import deque


def get_config(cfg ,args):
    cfgFile = args.configfile
    cfg.merge_from_file(cfgFile)


# set random seed
def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


# 学习速率调整策略
def lr_scheduler(optimizer, method):
    # LmbdaLR
    if method == 'multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 90, 120])
    elif method == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    elif method == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
    elif method == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', method)

    return scheduler

def BCELoss():
    loss_fnc = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.cuda.FloatTensor([1]))
    return loss_fnc

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()


# logger
def logger(cfg):
    logger_name = cfg.LOGGER.NAME
    log_save = os.path.join(cfg.SAVE.LOG_EXP,logger_name)
    mkdir(cfg.SAVE.LOG_EXP)
    # logger name
    logger = logging.getLogger(logger_name)
    # (debug, info, warning, error,critical)
    logger.setLevel(level=logging.INFO)

    if not logger.handlers:

        fh = logging.FileHandler(filename=log_save)
        formatter = logging.Formatter('%(asctime)s, %(name)s: %(filename)s: %(levelname)s: %(funcName)s: %(message)s',datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

    return logger

class HistoryBuffer:
    """The class tracks a series of values and provides access to the smoothed
    value over a window or the global average / sum of the sequence.
    Args:
        window_size (int): The maximal number of values that can
            be stored in the buffer. Defaults to 20.
    Example::
        >>> his_buf = HistoryBuffer()
        >>> his_buf.update(0.1)
        >>> his_buf.update(0.2)
        >>> his_buf.avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        self._history = deque(maxlen=window_size)
        self._count: int = 0
        self._sum: float = 0.0

    def update(self, value: float) -> None:
        """Add a new scalar value. If the length of queue exceeds ``window_size``,
        the oldest element will be removed from the queue.
        """
        self._history.append(value)
        self._count += 1
        self._sum += value

    @property
    def latest(self) -> float:
        """The latest value of the queue."""
        return self._history[-1]

    @property
    def avg(self) -> float:
        """The average over the window."""
        return np.mean(self._history)

    @property
    def global_avg(self) -> float:
        """The global average of the queue."""
        return self._sum / self._count

    @property
    def global_sum(self) -> float:
        """The global sum of the queue."""
        return self._sum




def print_table(epoch,loss,f1,iou,mask_acc,mask_pos_acc,lr,time):
    """
    :return:
    """
    tb = PrettyTable()
    tb.field_names = ['Epoch','Loss','F1','IoU','Mask_acc','Mask_pos_acc','Learning Rate',"Time (min)"]
    tb.add_row([epoch,loss,f1,iou,mask_acc,mask_pos_acc,lr,time])

    return tb

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)



def weights_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def load_checkpoints(model, checkpoint, optimizer,loadOptimizer):
    if checkpoint != None:
        model_dict = model.state_dict()
        pretrained_ckp = torch.load(checkpoint)
        # pretrained_dict = pretrained_ckp['state_dict']
        pretrained_dict = pretrained_ckp.copy()

        new_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys()}

        model_dict.update(new_dict)

        print("Total: {}, update:{}".format(len(pretrained_dict), len(new_dict)))

        model.load_state_dict(model_dict)
        print('loaded finished')

        if loadOptimizer == True:
            optimizer.load_state_dict(pretrained_ckp['optimizer'])
            print('loaded! optimizer')
        else:
            print("not loaded optimizer")

    else:
        print("No checkpoint is included")

    return model, optimizer