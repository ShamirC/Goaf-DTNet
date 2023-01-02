import os
import time
import numpy as np
import torch
import argparse
import yaml

from models.Goaf_DTNet import GoafDTNet
from mffdn_trainer import Trainer
from configs.defaults import _C as cfg
from utils.util import lr_scheduler,set_seed,mkdir,print_table, get_config, weights_init


def main():
    # set the seed
    set_seed(123)
    # configfile
    cfgFile = r"E:\cxm\Crack\GoafCrack_v6\configs\mffdn.yaml"
    cfg.merge_from_file(cfgFile)
    print(cfg)
    cfg.freeze()

    # network
    net = GoafDTNet(cfg)
    # print the amount of model parameters
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    # model.parameters()
    for param in net.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue  # total parameters
        if param.requires_grad:
            Trainable_params += mulValue  # trainable parameters
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

    # network initialization
    net.apply(weights_init)

    trainer = Trainer(cfg, net)
    # learning_scheduler = lr_scheduler(trainer.optimizer, cfg.TRAIN.SCHEDULER)
    learning_scheduler = trainer.lr_scheduler
    # save the configuration to disk
    with open(os.path.join(cfg.SAVE.LOG_EXP, "configuration.txt"), "w") as f:
        f.write(cfg.dump())

    # logger
    logger = trainer.logger
    # early stop
    early_stop = EarlyStopping(patience=5, verbose=True)
    # start training from this epoch
    start_epoch = trainer.start_epoch
    print("===================== Start Training Total Epoch %d=====================" % (cfg.TRAIN.EPOCH))
    for epoch in range(start_epoch + 1, cfg.TRAIN.EPOCH + 1):
        epoch_start_time = time.time()
        print("--------------- Training Iteration ---------------")
        trainer.train(epoch)
        train_loss = trainer.log_loss
        train_acc = trainer.log_acc

        # log epoch loss and metric value
        trainer.tensorboard.epoch_loss('Train', epoch, train_loss)
        trainer.tensorboard.criteria('Train', epoch, train_acc)

        # log the loss and metric information in a txt file
        logger.info(
            "[Train] Epoch %d/%d Segmentation Total Loss: %.3f, Extraction Total Loss: %.3f, Network Total Loss: %.3f, F1: %.3f, IoU: %.3f, Mask_acc: %.3f, Mask_pos_acc: %.3f"
            % (epoch, cfg.TRAIN.EPOCH, train_loss['seg_loss'], train_loss['extraction_loss'], train_loss['total_loss'],
               train_acc['f1'], train_acc['iou'], train_acc['mask_acc'], train_acc['mask_pos_acc']))

        #
        print("--------------- Validating Iteration ---------------")
        trainer.val(epoch)
        val_loss = trainer.log_loss
        val_acc = trainer.log_acc

        trainer.tensorboard.epoch_loss('Val', epoch, val_loss)
        trainer.tensorboard.criteria('Val', epoch, val_acc)

        logger.info(
            "[Val] Epoch %d/%d Segmentation Total Loss: %.3f, Extraction Total Loss: %.3f, Networkl Total Loss: %.3f, F1: %.2f, IoU: %.2f, Mask_acc: %.2f, Mask_pos_acc: %.2f"
            % (epoch, cfg.TRAIN.EPOCH, val_loss['seg_loss'], val_loss['extraction_loss'], val_loss['total_loss'],
               val_acc['f1'], val_acc['iou'], val_acc['mask_acc'], val_acc['mask_pos_acc']))

        learning_scheduler.step()
        # early stopping，save the weights
        early_stop(val_loss['extraction_loss'], val_acc['iou'], net, trainer.optimizer, epoch, learning_scheduler)
        # current learning of the epoch
        current_lr = trainer.optimizer.param_groups[0]['lr']
        logger.info("[Train] Epoch %d/%d LR: %.3f" % (epoch, cfg.TRAIN.EPOCH, current_lr))

        # time spent for an epoch
        epoch_end_time = time.time() - epoch_start_time
        # print a table contain the information about loss , metric, consumed time
        tb = print_table(epoch, round(val_loss['total_loss'],3), round(val_acc['f1'],3), round(val_acc['iou'],3), round(val_acc['mask_acc'],3),
                         round(val_acc['mask_pos_acc'], 3), current_lr, round(epoch_end_time / 60, 2))
        print(tb)
        # update the learning rate

        if early_stop.early_stop == True:
            break


class EarlyStopping():
    def __init__(self,patience=5, verbose=False,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss,val_acc, model, optimizer,epoch, lr_scheduler):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,val_acc,model,optimizer,epoch, lr_scheduler)
        # if the accuracy decreased，self.counter +1，else self.counter >= patience，self.early_stop=True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,val_acc,model,optimizer,epoch, lr_scheduler)
            self.counter = 0

    def save_checkpoint(self,val_loss,val_acc,model,optimizer,epoch,lr_scheduler):
        # model name： ModelName_loss_acc_time_epoch.pth
        """
        """
        save_dir = cfg.SAVE.LOG_EXP
        mkdir(save_dir)
        ckp_name = cfg.MODEL.NAME + '_loss_%.3f_'%(val_loss) + '%.2f_f1'%(val_acc) + "%s_"%(time.strftime("%m-%d-%H-%M-%S", time.localtime())) + "epoch%d"%(epoch)+ ".pth"
        path = os.path.join(save_dir,ckp_name)
        checkpoint = {
            "epoch": epoch,  # epoch
            "state_dict": model.state_dict(),  # model parameters
            "optimizer": optimizer.state_dict(),  # optimizer parameters
            "best_loss": val_loss, # loss
            "lr_scheduler": lr_scheduler, # learning scheduler
        }
        if self.verbose:
            print(f'acc increased ({self.val_acc_min:.3f} --> {val_acc:.3f}).  Saving model ...')
        torch.save(checkpoint, path)
        self.val_acc_min = val_acc


if __name__ == "__main__":
    main()