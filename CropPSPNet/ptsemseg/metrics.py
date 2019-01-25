# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        hist_sum = hist.sum(axis=1)
        for k in range(len(hist_sum)):
            if hist_sum[k] == 0:
                hist_sum[k] = 1.0e-6

        precision_cls = np.diag(hist) / hist.sum(axis=1)
        recall_cls = np.diag(hist) / hist.sum(axis=0)
        mean_precision = np.mean(precision_cls)
        mean_recall = np.mean(recall_cls)
        
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))


        Po = acc
        Pe_0 = (hist.sum(axis=1)[0]/hist.sum())*(hist.sum(axis=0)[0]/hist.sum())
        Pe_1 = (hist.sum(axis=1)[1]/hist.sum())*(hist.sum(axis=0)[1]/hist.sum())
        Pe_2 = (hist.sum(axis=1)[2]/hist.sum())*(hist.sum(axis=0)[2]/hist.sum())
        Pe = Pe_0 + Pe_1 + Pe_2
        Kappa = (Po - Pe)/(1.0 - Pe)

        print('hist: {}'.format(hist))
        print('total_X (row): {}'.format(hist.sum(axis=1)))
        print('total_Y (col): {}'.format(hist.sum(axis=0)))
        print('total_all: {}'.format(hist.sum()))
        print('UA(%): {}'.format(100*precision_cls))
        print('PA(%): {}'.format(100*recall_cls))
        print('OA(%): {}'.format(100*acc))
        print('Kappa(%): {}'.format(100*Kappa))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "Mean Precision : \t": mean_precision,
                "Mean Recall : \t": mean_recall,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

