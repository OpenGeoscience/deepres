"""
Training routines for cropnet models using Jon Crall's NetHarn
"""

import netharn as nh
import numpy as np
import random
import torch
import torchvision as tv
import ubelt as ub

from model import CropNetFCAE
from netharn.metrics import clf_report
from torchvision import transforms


class CropNetFCAE_FitHarn(nh.FitHarn):
    def __init__(harn, *args, **kw):
        super(CropNetFCAE_FitHarn, self).__init__(*args, **kw)
        harn.batch_confusions = []

    def run_batch(harn, batch):
        """
        Custom function to compute the output of a batch and its loss.
        """
        inputs, labels = batch
        output = harn.model(*inputs)
        label = labels[0]
        loss = harn.criterion(output, label)
        outputs = [output]
        return outputs, loss

    # TODO this clearly needs to be amended for AE
    def on_batch(harn, batch, outputs, loss):
        inputs, labels = batch
        label = labels[0]
        output = outputs[0]

        y_pred = output.data.max(dim=1)[1].cpu().numpy()
        y_true = label.data.cpu().numpy()
        probs = output.data.cpu().numpy()

        harn.batch_confusions.append((y_true, y_pred, probs))

    # TODO this clearly needs to be amended for AE
    def on_epoch(harn):
        """
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 2, 2])
        y_pred = np.array([1, 1, 1, 2, 1, 0, 0, 0, 2, 2])
        all_labels = np.array([0, 1, 2])
        """

        dset = harn.datasets[harn.current_tag]
        target_names = dset.class_names

        all_trues, all_preds, all_probs = zip(*harn.batch_confusions)

        probs = np.vstack(all_probs)
        y_true = np.hstack(all_trues)
        y_pred = np.hstack(all_preds)

        # Compute multiclass metrics (new way!)
        report = clf_report.ovr_classification_report(
            y_true, probs, target_names=target_names, metrics=[
                'auc', 'ap', 'mcc', 'brier'
            ])
        #print(ub.repr2(report))

        # percent error really isn't a great metric, but its standard.
        errors = (y_true != y_pred)
        percent_error = errors.mean() * 100
        # cfsn = confusion_matrix(y_true, y_pred, labels=all_labels)

        # global_acc = global_accuracy_from_confusion(cfsn)
        # class_acc = class_accuracy_from_confusion(cfsn)

        metrics_dict = ub.odict()
        metrics_dict['ave_brier'] = report['ave']['brier']
        metrics_dict['ave_mcc'] = report['ave']['mcc']
        metrics_dict['ave_auc'] = report['ave']['auc']
        metrics_dict['ave_ap'] = report['ave']['ap']
        metrics_dict['percent_error'] = percent_error
        metrics_dict['acc'] = 1 - percent_error

        harn.batch_confusions.clear()
        return metrics_dict


def train():
    np.random.seed(1031726816 % 4294967295)
    torch.manual_seed(137852547 % 4294967295)
    random.seed(2497950049 % 4294967295)

    # batch_size = int(ub.argval('--batch_size', default=128))
    batch_size = int(ub.argval('--batch_size', default=64))
    workers = int(ub.argval('--workers', default=6))
    model_key = ub.argval('--model', default="CropNetFCAE")
    xpu = nh.XPU.cast("gpu")

    lr = 0.001

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    workdir = ub.ensure_app_cache_dir('netharn')

    datasets = {
        'train': torchvision.datasets.CIFAR10(root=workdir, train=True,
                                              download=True,
                                              transform=transform_train),
        'test': torchvision.datasets.CIFAR10(root=workdir, train=False,
                                             download=True,
                                             transform=transform_test),
    }

    # For some reason the torchvision objects dont have the label names
    CIFAR10_CLASSNAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck',
    ]
    datasets['train'].class_names = CIFAR10_CLASSNAMES
    datasets['test'].class_names = CIFAR10_CLASSNAMES

    n_classes = 10  # hacked in
    loaders = {
        key: torch.utils.data.DataLoader(dset, shuffle=key == 'train',
                                         num_workers=workers,
                                         batch_size=batch_size,
                                         pin_memory=True)
        for key, dset in datasets.items()
    }

    if workers > 0:
        import cv2
        cv2.setNumThreads(0)

    initializer_ = (nh.initializers.KaimingNormal, {'param': 0, 'mode': 'fan_in'})
    # initializer_ = (initializers.LSUV, {})

    available_models = {
            "CropNetFCAE": (CropNetFCAE, {
                "chip_size": 19,
                "bneck_size" : 3,
                }),
    }
    model_ = available_models[model_key]

    hyper = nh.HyperParams(
        datasets=datasets,
        nice='cifar10_' + model_key,
        loaders=loaders,
        workdir=workdir,
        xpu=xpu,
        model=model_,
        optimizer=(torch.optim.SGD, {
            'lr': lr,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'nesterov': True,
        }),
        scheduler=(nh.schedulers.ListedLR, {
            'points': {
                0: lr,
                150: lr * 0.1,
                250: lr * 0.01,
            },
            'interpolate': False
        }),
        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'patience': 350,
            'max_epoch': 350,
        }),
        initializer=initializer_,
        criterion=(torch.nn.CrossEntropyLoss, {}),
        # Specify anything else that is special about your hyperparams here
        # Especially if you make a custom_batch_runner
        # TODO: type of augmentation as a parameter dependency
        # augment=str(datasets['train'].augmenter),
        # other=ub.dict_union({
        #     # 'colorspace': datasets['train'].output_colorspace,
        # }, datasets['train'].center_inputs.__dict__),
    )
    harn = CIFAR_FitHarn(hyper=hyper)
    harn.initialize()
    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        python examples/cifar.py --gpu=0 --model=densenet121
        python examples/cifar.py --gpu=0 --model=resnet50
        # Train on two GPUs with a larger batch size
        python examples/cifar.py --model=dpn92 --batch_size=256 --gpu=0,1
    """
    train()

