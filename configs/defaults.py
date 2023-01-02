from yacs.config import CfgNode as CN
import time
import random


_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = 'cuda'    # train on 'cuda' or ’cpu'
_C.MODEL.CH_IN = 3  # model input channels
_C.MODEL.NUM_CLASSES = 1 # class number
_C.MODEL.ASPP = True # ASPP module
_C.MODEL.SGFM = True # add SGFM module
_C.MODEL.MFFM = True # add MFFM module

_C.MODEL.PRETRAINED = True
# ------------------------------------------------------------

_C.DATA = CN()
_C.DATA.DATA_LIST_TRAIN = r"./data/train.txt"
_C.DATA.DATA_LIST_VAL = r"./data/valid.txt"

#-------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.LR = 1e-3
_C.TRAIN.DEVICE = 'cuda'
_C.TRAIN.PRINT_FREQ = 50
_C.TRAIN.VAL_EVERY = 100
_C.TRAIN.EPOCH = 100
_C.TRAIN.EXP_NUM = 1
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.SCHEDULER = "step"     # "cosine","step", "plateau"
_C.TRAIN.LOSS = "BCE"
_C.TRAIN.RESUME = False  # resume training process

_C.SAVE = CN()
# tensorboard log file name
_C.SAVE.LOG_NAME = 'runs/%s-%05d' %(time.strftime("%m-%d-%H-%M-%S", time.localtime()),random.randint(0, 100))
# create every single experiment a new file
_C.SAVE.LOG_EXP = 'E:/chenximing/Cracks/GoafCrack_v2/logs/%s/exp%d/' %(_C.MODEL.NAME, _C.TRAIN.EXP_NUM )

#
_C.LOGGER = CN()
_C.LOGGER.NAME = _C.MODEL.NAME + "_log.txt"

#
_C.TASK = CN()

