import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

from data.transforms import augCompose, RandomBlur, RandomColorJitter,RandomFlip
from utils.util import logger, BCELoss, HistoryBuffer, load_checkpoints, BinaryFocalLoss, lr_scheduler
from utils.loss import get_lossfunc
from utils.tensorboards import CrackTensorBoard, Metrics
from data.dataset import GoafCrackSeg


class Trainer():
    def __init__(self, cfg, model):
        super(Trainer, self).__init__()

        self.device = cfg.TRAIN.DEVICE
        self.cfg = cfg
        self.model = model.to(self.device)

        self.transforms = augCompose(transforms=[[RandomFlip, 0.5], [RandomBlur, 0.5], [RandomColorJitter, 0.5]])
        self.optimizer = torch.optim.Adam(model.parameters(), cfg.TRAIN.LR)
        self.lr_scheduler = lr_scheduler(self.optimizer, cfg.TRAIN.SCHEDULER)
        self.loss = get_lossfunc(cfg.TRAIN.LOSS)
        # self.loss = BinaryFocalLoss()
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.start_epoch = 0

        self.train_dataset = GoafCrackSeg(self.cfg, is_train=True, transforms=self.transforms)
        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, drop_last=True, num_workers=8)
        self.val_dataset = GoafCrackSeg(self.cfg, is_train=False, transforms=None)
        self.val_dataloader = DataLoader(self.val_dataset, self.batch_size, shuffle=True, drop_last=True, num_workers=8)

        self.tensorboard = CrackTensorBoard(cfg)
        self.logger = logger(cfg)
        self.print_freq = cfg.TRAIN.PRINT_FREQ
        self.epoch = cfg.TRAIN.EPOCH

        # load pre-trained model weights
        if cfg.MODEL.PRETRAINED == True:
            self.model, self.optimizer = load_checkpoints(self.model, self.resnet_ckp, self.optimizer, False)
            self.logger.info("Pretrained Weights Loaded")
        # resume model training process
        if cfg.TRAIN.RESUME:
            path_checkpoint = "./checkpoints/xxx.pth"  # last saved checkpoint
            checkpoint = torch.load(path_checkpoint)  # load checkpoint
            model.load_state_dict(checkpoint["state_dict"])  # load model parameters
            self.optimizer.load_state_dict(checkpoint["optimizer"])  #load optimizer parameters
            self.start_epoch = checkpoint["epoch"]  # resume in this epoch
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.train_iter_count = 0
        self.val_iter_count = 0

        self.log_loss = {}
        self.log_acc = {}


    def train(self, epoch):
        self.model.train()

        segloss_buffer = HistoryBuffer()
        extractionloss_buffer = HistoryBuffer()
        totalloss_buffer = HistoryBuffer()
        mask_acc_buffer = HistoryBuffer()
        mask_pos_acc_buffer = HistoryBuffer()
        f1_buffer = HistoryBuffer()
        iou_buffer = HistoryBuffer()

        for idx, data in enumerate(self.train_dataloader):
            self.train_iter_count += 1
            batch_loss = {}

            image, seg_label, line_label = data[0], data[1], data[2]
            image, seg_label, line_label = image.type(torch.cuda.FloatTensor).to(self.device), seg_label.type(
                torch.cuda.FloatTensor).to(self.device), line_label.type(torch.cuda.FloatTensor).to(self.device)

            out = self.model(image)

            segmentation_out, extraction_out = out[0], out[1]
            # segmentation head output
            seg_end = segmentation_out['end_out']
            seg_ds1, seg_ds2, seg_ds3, seg_ds4 = [segmentation_out["ds_out{}".format(str(i))] for i in range(1, 5)]
            seg_ds = [seg_ds1, seg_ds2, seg_ds3, seg_ds4]

            # extaction head output
            extraction_end = extraction_out["end_out"]
            extraction_ds1, extraction_ds2, extraction_ds3, extraction_ds4 = [extraction_out["ds_out{}".format(str(i))] for i in range(1, 5)]
            extraction_ds = [extraction_ds1, extraction_ds2, extraction_ds3, extraction_ds4]

            # compute segmentation loss
            seg_loss_end = self.loss(seg_end.squeeze(1), seg_label)
            seg_loss_ds1, seg_loss_ds2, seg_loss_ds3, seg_loss_ds4 = [self.loss(i.squeeze(1), seg_label) for i in seg_ds]

            # compute extraction loss
            extraction_loss_end = self.loss(extraction_end.squeeze(1), line_label)
            extraction_ds1, extraction_ds2, extraction_ds3, extraction_ds4 = [self.loss(i.squeeze(1), line_label) for i in extraction_ds]

            # compute total loss of segmentation head
            segmentation_total_loss = seg_loss_end+seg_loss_ds1+seg_loss_ds2+seg_loss_ds3+seg_loss_ds4
            # compute total loss of extaction head
            extraction_total_loss = extraction_loss_end+extraction_ds1+extraction_ds2+extraction_ds3+extraction_ds4

            # comput the total loss of the network
            total_loss = segmentation_total_loss + extraction_total_loss

            # log batch loss
            batch_loss["seg_total_loss"] = segmentation_total_loss
            batch_loss["extraction_total_loss"] = extraction_total_loss

            segloss_buffer.update(segmentation_total_loss.item())
            extractionloss_buffer.update(extraction_total_loss.item())
            totalloss_buffer.update(total_loss.item())

            # compute the metric for both segmentation head
            segmentation_metric = Metrics(seg_end, seg_label)
            f1 = segmentation_metric.f1_score()
            iou = segmentation_metric.iou()

            f1_buffer.update(f1)
            iou_buffer.update(iou)

            # compute the metric for extraction head
            acc = self.mask_acc(extraction_end, line_label)
            mask_pos_acc = acc["mask_pos_acc"]
            mask_acc = acc["mask_acc"]
            mask_acc_buffer.update(mask_acc)
            mask_pos_acc_buffer.update(mask_pos_acc)

            #
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # tensorboard to log the batch loss
            self.tensorboard.batch_loss("Train", self.train_iter_count, batch_loss)

            # using logging to write the loss and metric information to a txt file
            if (idx + 1) % self.print_freq == 0:
                self.logger.info(
                    '[Train] batch %s Segmentation Total Loss: %.3f ; Extraction Total Loss: %.3f ; Network Total Loss: %.3f; '
                    'F1: %.2f, IoU: %.2f; '
                    'Mask_pos_acc: %.2f; Mask_acc: %.2f' % (
                        idx + 1, segloss_buffer.avg, extractionloss_buffer.avg, totalloss_buffer.avg,
                        f1_buffer.avg, iou_buffer.avg, mask_pos_acc_buffer.avg, mask_acc_buffer.avg)
                )

                # batch output of the  for visualization
                batch_fmap = {
                    "segmentation_prediction": torch.sigmoid(seg_end.clone()),
                    "segmentation_label": seg_label.unsqueeze(1),
                    "extraction_prediction": torch.sigmoid(extraction_end.clone()),
                    "extraction_label": line_label.unsqueeze(1),
                }
                self.tensorboard.fmap("Train", self.train_iter_count, batch_fmap)

                print("Training : Epoch %d/%d Batch %d/%d; Total Loss: %.3f "
                      % (epoch, self.epoch, idx+1, len(self.train_dataloader), totalloss_buffer.avg))

        # one epoch ends
        # log the global average loss of one epoch in the self.lo_loss dictionary
        self.log_loss["seg_loss"] = segloss_buffer.global_avg
        self.log_loss["extraction_loss"] = extractionloss_buffer.global_avg
        self.log_loss["total_loss"] = totalloss_buffer.global_avg

        # log the global average metrics of one epoch in the self.lo_loss dictionary
        self.log_acc['f1'] = f1_buffer.global_avg
        self.log_acc['iou'] = iou_buffer.global_avg
        self.log_acc['mask_acc'] = mask_acc_buffer.global_avg
        self.log_acc['mask_pos_acc'] = mask_pos_acc_buffer.global_avg

    def val(self, epoch):
        self.model.eval()

        self.log_loss = {}
        self.log_acc = {}

        segloss_buffer = HistoryBuffer()
        extractionloss_buffer = HistoryBuffer()
        totalloss_buffer = HistoryBuffer()
        mask_acc_buffer = HistoryBuffer()
        mask_pos_acc_buffer = HistoryBuffer()
        f1_buffer = HistoryBuffer()
        iou_buffer = HistoryBuffer()

        with torch.no_grad():
            for idx, data in enumerate(self.val_dataloader):
                self.val_iter_count += 1
                batch_loss = {}

                image, seg_label, line_label = data[0], data[1], data[2]
                image, seg_label, line_label = image.type(torch.cuda.FloatTensor).to(self.device), seg_label.type(
                    torch.cuda.FloatTensor).to(self.device), line_label.type(torch.cuda.FloatTensor).to(self.device)

                out = self.model(image)

                segmentation_out, extraction_out = out[0], out[1]

                # segmentation head output
                seg_end = segmentation_out['end_out']
                seg_ds1, seg_ds2, seg_ds3, seg_ds4 = [segmentation_out["ds_out{}".format(str(i))] for i in range(1, 5)]
                seg_ds = [seg_ds1, seg_ds2, seg_ds3, seg_ds4]

                # extaction head output
                extraction_end = extraction_out["end_out"]
                extraction_ds1, extraction_ds2, extraction_ds3, extraction_ds4 = [
                    extraction_out["ds_out{}".format(str(i))] for i in range(1, 5)]
                extraction_ds = [extraction_ds1, extraction_ds2, extraction_ds3, extraction_ds4]

                # compute segmentation loss
                seg_loss_end = self.loss(seg_end.squeeze(1), seg_label)
                seg_loss_ds1, seg_loss_ds2, seg_loss_ds3, seg_loss_ds4 = [self.loss(i.squeeze(1), seg_label) for i in seg_ds]

                # compute extraction loss
                extraction_loss_end = self.loss(extraction_end.squeeze(1), line_label)
                extraction_ds1, extraction_ds2, extraction_ds3, extraction_ds4 = [self.loss(i.squeeze(1), line_label) for i in
                                                                                  extraction_ds]

                # compute total loss of segmentation head
                segmentation_total_loss = seg_loss_end + seg_loss_ds1 + seg_loss_ds2 + seg_loss_ds3 + seg_loss_ds4
                # compute total loss of extaction head
                extraction_total_loss = extraction_loss_end + extraction_ds1 + extraction_ds2 + extraction_ds3 + extraction_ds4

                # comput the total loss of the network
                total_loss = segmentation_total_loss + extraction_total_loss

                # log batch loss
                batch_loss["seg_total_loss"] = segmentation_total_loss
                batch_loss["extraction_total_loss"] = extraction_total_loss

                segloss_buffer.update(segmentation_total_loss.item())
                extractionloss_buffer.update(extraction_total_loss.item())
                totalloss_buffer.update(total_loss.item())

                # compute the metric for both segmentation head
                segmentation_metric = Metrics(seg_end, seg_label)
                f1 = segmentation_metric.f1_score()
                iou = segmentation_metric.iou()

                f1_buffer.update(f1)
                iou_buffer.update(iou)

                # compute the metric for extraction head
                acc = self.mask_acc(extraction_end, line_label)
                mask_pos_acc = acc["mask_pos_acc"]
                mask_acc = acc["mask_acc"]
                mask_acc_buffer.update(mask_acc)
                mask_pos_acc_buffer.update(mask_pos_acc)

                # tensorboard to log the batch loss
                self.tensorboard.batch_loss("Val", self.val_iter_count, batch_loss)

                # using logging to write the loss and metric information to a txt file
                if (idx + 1) % self.print_freq == 0:
                    self.logger.info(
                        '[Val] batch %s Segmentation Total Loss: %.3f ; Extraction Total Loss: %.3f ; Network Total Loss: %.3f; '
                        'F1: %.2f, IoU: %.2f; '
                        'Mask_pos_acc: %.2f; Mask_acc: %.2f' % (
                            idx + 1, segloss_buffer.avg, extractionloss_buffer.avg, totalloss_buffer.avg,
                            f1_buffer.avg, iou_buffer.avg, mask_pos_acc_buffer.avg, mask_acc_buffer.avg)
                    )

                    # batch output of the  for visualization
                    batch_fmap = {
                        "segmentation_prediction": torch.sigmoid(seg_end.clone()),
                        "segmentation_label": seg_label.unsqueeze(1),
                        "extraction_prediction": torch.sigmoid(extraction_end.clone()),
                        "extraction_label": line_label.unsqueeze(1),
                    }
                    self.tensorboard.fmap("Val", self.val_iter_count, batch_fmap)

                    print("Validating : Epoch %d/%d Batch %d/%d; Total Loss: %.3f "
                          % (epoch, self.epoch, idx + 1, len(self.val_dataloader), totalloss_buffer.avg))

        # one epoch ends
        # log the global average loss of one epoch in the self.lo_loss dictionary
        self.log_loss["seg_loss"] = segloss_buffer.global_avg
        self.log_loss["extraction_loss"] = extractionloss_buffer.global_avg
        self.log_loss["total_loss"] = totalloss_buffer.global_avg

        # log the global average metrics of one epoch in the self.lo_loss dictionary
        self.log_acc['f1'] = f1_buffer.global_avg
        self.log_acc['iou'] = iou_buffer.global_avg
        self.log_acc['mask_acc'] = mask_acc_buffer.global_avg
        self.log_acc['mask_pos_acc'] = mask_pos_acc_buffer.global_avg

        # end

    def mask_acc(self,pred,label):
        preds = F.sigmoid(pred).clone()
        preds = preds.squeeze(1).cpu().detach().numpy()

        preds = (preds > 0.5).astype(np.uint8)

        labels = label.clone()
        labels = labels.cpu().detach().numpy().astype(np.uint8)

        assert labels.shape == preds.shape

        eps = 1e-6
        mask_acc = np.sum(preds==labels)/(labels.size+eps)
        mask_pos_acc = np.sum((preds == 1) & (labels == 1))/ (((labels==1).sum())+eps)
        mask_neg_acc = np.sum((preds == 0) & (labels == 0))/ (((labels==0).sum())+eps)

        log_acc = {
            "mask_acc": mask_acc,
            "mask_pos_acc": mask_pos_acc,
            "mask_neg_acc": mask_neg_acc,
            }

        return log_acc

