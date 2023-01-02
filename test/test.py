
from models.Goaf_DTNet import GoafDTNet
import numpy as np
from torch.utils import data
import glob
import os
import cv2
import torch
from skimage import io
import csv
from scipy import io as scio
from configs.defaults import _C as cfg


def test_forward():
    """
    implement forward propagation using test image
    and save the prediction
    """

    image_dtaroot = r"./data/Goaf_Cracks/kq13_dom"
    label_dataroot = r"./data/Goaf_Cracks/kq13_label_seg"

    # test_dataset = TestDataset(args.image_dataroot, args.label_dataroot)
    test_dataset = TestDataset(image_dtaroot, label_dataroot)
    test_dataloader = data.DataLoader(test_dataset, 1)

    # directory to save the output of the test procedure
    # result = args.result
    result = r"../results/mffdn_aspp_sgfm_line"
    if not os.path.exists(result):
        os.mkdir(result)

    # path to save the csv file recording evaluation results
    # csv_name = args.csv
    csv_name = "prf_test.csv"
    csv_path = os.path.join(result, csv_name)

    prf_results = prf_metrics_multi_thresh(test_dataloader)
    with open(csv_path, "a+") as f:
        csv_writer = csv.writer(f)
        for prf in prf_results:
            csv_writer.writerow(prf)

    for prf in prf_results:
        print(prf)


cfgFile = r'../configs/mffdn.yaml'
cfg.merge_from_file(cfgFile)
print(cfg)

def prf_metrics_multi_thresh(test_dataloader, thresh_step=0.01):
    final_accuracy_all = []  # list used to store the precision, recall, f1-score, iou in different threshold,
                            # exp. [[0.01,precision,recall,f-score,iou], [0.02,precision,recall, f-score, iou],...,..., [0.99, precision,recall,iou]
    net = GoafDTNet(cfg).cuda()
    net.load_state_dict(torch.load(r"./checkpoints/Goaf_DTNet.pth")["state_dict"])
    net.eval()
    eps = 1e-6
    for thresh in np.arange(0.0, 1.0, thresh_step):
        print("current threshold", thresh)
        statistics = []
        for img, lab, img_name in test_dataloader:
            img, lab = img.cuda(), lab.cuda()
            pred = net(img)
            pred_seg = pred[0]["end_out"]
            pred_line = pred[1]["end_out"]

            # save the prediction map derived from threshold of 0.5
            if thresh == 0.5:
                pth = r"../results/mffdn/seg"
                save_pred(thresh, pred_seg, img_name, pth)
                pth = r"../results/mffdn/extract"
                save_pred(thresh, pred_line, img_name, pth)

            # calculate precision, recall, f-score, iou under this threshold, store the results in a list;
            # only for the segmentation task
            pred_seg= torch.sigmoid(pred_seg)
            pred_seg = pred_seg.detach().cpu().squeeze().numpy()
            pred_seg[pred_seg >= thresh] = 1
            pred_seg[pred_seg < thresh] = 0

            statistics.append(get_statistics(pred_seg, lab.detach().cpu().numpy()))

        # sum up the TPs, FPs, FNs of the whole dataset under the threshold
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        # calculate precision
        precision = 1.0 if tp == 0 and fp == 0 else tp/(tp+fp+eps)
        # calculate recall
        recall = tp/(tp+fn+eps)
        # calculate f-score
        f = 2*precision*recall / (precision + recall + eps)
        # calculate iou
        iou = tp / (tp + fp + fn + eps)
        final_accuracy_all.append([thresh, precision, recall, f, iou])
        print("precision: {}, recall: {}, f 得分： {}, IoU: {}".format(precision, recall, f, iou))

    return final_accuracy_all  # [[threshold0, precision, recall, f-score, iou], [threshold, precision, recall, f-score, iou] ... , ...]


def save_pred(thresh, pred, img_name, pth):
    if not os.path.exists(pth):
        os.mkdir(pth)
    pred_name = os.path.join(pth, img_name[0] + ".png")
    # save the results in png format
    pred = torch.sigmoid(pred)
    pred = pred.detach().cpu().squeeze().numpy()
    pred_ = np.zeros((256,256), dtype=np.uint8)
    pred_[pred < thresh] = 1
    io.imsave(pred_name, (pred_*255).astype(np.uint8))


def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]


class TestDataset(data.Dataset):
    """
    load test image and label
    """
    def __init__(self, image_dataroot, label_dataroot):
        self.image_root = image_dataroot
        self.label_root = label_dataroot
        self.image_list = glob.glob(os.path.join(self.image_root, "*.{}".format("png")))
        self.label_list = glob.glob(os.path.join(self.label_root, "*.{}".format("bmp")))
        assert len(self.image_list) == len(self.label_list)

        self.image_name = [image.split("\\")[-1].split(".")[0] for image in self.image_list]

    def __getitem__(self, item):
        im = cv2.cvtColor(cv2.imread(self.image_list[item]), cv2.COLOR_BGR2RGB)
        im = im.transpose((2,0,1)).astype(np.float32)
        lab = cv2.imread(self.label_list[item], cv2.IMREAD_GRAYSCALE)

        return im, lab/255.0, self.image_name[item]     # return (image, label, image_name)

    def __len__(self):
        return len(self.image_list)


def mk_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


if __name__ == "__main__":
    test_forward()

