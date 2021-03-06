"""
This is the trainer for the autoencoder.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# pytorch imports
from torch.autograd import Variable

# ml_utils imports
from pyt_utils.trainer_base import TrainerBase
from torchvision.utils import save_image

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

class AETrainer(TrainerBase):
    def __init__(self, input_size=19, sampler_batch_size=32, **kwargs):
        super(AETrainer, self).__init__(**kwargs)
        self._input_size = input_size
        self._sampler_batch_size = sampler_batch_size

    def _criterion(self, model_output, x):
        recon_x,mu,logvar = model_output
        size_sq = self._input_size * self._input_size
        recon_x = recon_x.view(-1, size_sq)
        x = x.view(-1, size_sq)
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (BCE + KLD, BCE, KLD)


    def _get_optimizer(self):
        return super(AETrainer, self)._get_optimizer()

    def _write_batch(self, *args):
        super(AETrainer, self)._write_batch(*args)

    def _write_epoch(self, *args):
        epoch,test_batch_ct = args
        img_size = self._model.get_input_size()
        dataset = self.get_test_loader().dataset
        inc = len(dataset) // self._sampler_batch_size
        inputs = []
        targets = []
        for i in range(self._sampler_batch_size):
            inp,target = dataset[i*inc]
            inputs.append(inp)
            targets.append(target)
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        self._model.eval()
        outputs = self._model( Variable(inputs).cuda() )[0]
        output_imgs = [o.data.cpu() for o in torch.squeeze(outputs)]
        output_imgs = torch.stack(output_imgs).view(self._sampler_batch_size,
                1, img_size, img_size)

        comparisons = torch.cat((Variable(inputs), Variable(output_imgs),
            Variable(targets)))
        save_image(comparisons, pj(self._samples_dir, "comparisons_%03d.png" \
                % (epoch)))

def class_sampler(epoch, trainer):
    model = trainer.get_model()
    model.eval()
    dataset = trainer.get_train_loader().dataset # TODO should be a test image
    data_loader = trainer.get_train_loader()
    output_img = np.zeros((500,500))
    cats_dict = dataset.get_cats_dict()
    inv_cats_dict = OrderedDict()
    for k,v in cats_dict.items():
        inv_cats_dict[v] = k
    inv_cats_dict[0] = 0
    output_img = []
    for batch_idx,(data,_) in enumerate(data_loader):
        data = Variable(data).cuda()
        outputs = model(data)
        for d in np.array( outputs.cpu().data ):
            output_img.append( inv_cats_dict[ np.argmax(d) ] )
    output_img = np.resize(np.array(output_img), (500,500))
    output_img = torch.Tensor(output_img) / 256.0

    save_image(output_img, pj(trainer._samples_dir, "classifications_%03d.png" \
            % (epoch)))



