"""
The RN-VAE for encoding crop data as RGB values
"""

import argparse
import torch
import torch.nn as nn
import torchvision as tv

from collections import OrderedDict
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


g_DEBUG = False

def load_ae_model(model_path, model_name, is_train=False, **kwargs):
    if model_name=="CropNetFCAE":
        model = CropNetFCAE(**kwargs)
    else:
        raise RuntimeError("Model %s not recognized" % (model_name))
    model.load_state_dict( torch.load(model_path) )
    if is_train:
        model = model.cuda().train()
    else:
        model = model.cuda().eval()
    return model

class CropNetFCAE(nn.Module):
    def __init__(self, chip_size=19, bneck_size=3):
        super(CropNetFCAE, self).__init__()

        self._bneck_size = bneck_size
        self._chip_size = chip_size
        self._name = "cropnetfcae"

        chip_size_sq = chip_size * chip_size

    

        self.fc1 = nn.Linear(chip_size_sq, 100)
        self.fc2 = nn.Linear(100, 20)
        self.bneck1 = nn.Linear(20, self._bneck_size)
        self.bneck2 = nn.Linear(20, self._bneck_size)
        self.fc3 = nn.Linear(self._bneck_size, 20)
        self.fc4 = nn.Linear(20, 100)
        self.fc5 = nn.Linear(100, chip_size_sq)

    def forward(self, x):
        mu,logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        out = self._decode(z)
        return out,mu,logvar

    def get_features(self, x):
        mu,_ = self._encode(x)
        return mu

    def get_input_size(self): # TODO Create a base class with this method, 
            # put in pyt_utils/modelbase.py
        return self._chip_size

    def get_name(self):
        return self._name

    def _decode(self, z):
        z = F.relu( self.fc3(z) )
        z = F.relu( self.fc4(z) )
        z = torch.sigmoid( self.fc5(z) )
        return z

    def _encode(self, x):
        x = x.view(-1, self._chip_size * self._chip_size)
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        return self.bneck1(x), self.bneck2(x)

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable( std.data.new(std.size()).normal_() )
                # !!! TODO, want a uniform variable here??
            z = eps.mul(std).add_(mu)
        else:
            z = mu
        return z


def conv_block_W(W, stride=2, padding=0, bias=False, is_transpose=False,
        output_padding=0, has_relu=True):
    if type(W) != nn.parameter.Parameter:
        raise RuntimeError("Incorrect type, %s given, must be %s" \
                % (type(W), nn.parameter.Parameter))

    if is_transpose:
        c_out = W.shape[1]
        c_in = W.shape[0]
        W.requires_grad = False
    else:
        c_out = W.shape[0]
        c_in = W.shape[1]
    k_sz = (W.shape[2], W.shape[3])
    if g_DEBUG: print(W.shape)

    if is_transpose:
        conv2d = nn.ConvTranspose2d(c_in, c_out, k_sz, stride=stride, 
                padding=padding, output_padding=output_padding, bias=bias)
    else:
        conv2d = nn.Conv2d(c_in, c_out, k_sz, stride=stride, padding=padding, 
                bias=bias)
    conv2d.weight = W
    conv2d.reset_parameters()
    if has_relu:
        conv_block = nn.Sequential(\
                conv2d,
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )
    else:
        conv_block = nn.Sequential(\
                conv2d,
                nn.BatchNorm2d(c_out),
            )
    return conv_block

def conv_seq(Ws, is_transpose):
    d = OrderedDict()
    N = len(Ws)
    for i in range(N):
        if is_transpose:
            idx = N - i - 1
            name = "deconv%d" % (i)
            output_padding = 1
            has_relu = False if i==N-1 else True
        else:
            idx = i
            name = "conv%d" % (i)
            output_padding = 0
            has_relu = True
        W = Ws[idx]
        padding = W.shape[2] // 2
        d[name] = conv_block_W(W, padding=padding, is_transpose=is_transpose,
                output_padding=output_padding, has_relu=has_relu)
    seq = nn.Sequential(d) 
    return seq

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, c_out, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(c_in, c_out, stride)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(c_out, c_out)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CropNetAE(nn.Module):
    def __init__(self, chip_size=32, num_rn_blocks=2, conv_per_rn=2,
            base_nchans=64, bneck_size=3, share_weights=True):
        super().__init__()

        self._base_nchans = base_nchans
        self._bneck_size = bneck_size
        self._chip_size = chip_size
        self._conv_per_rn = conv_per_rn
        self._decoder = None
        self._encoder = None
        self._num_rn_blocks = num_rn_blocks

#        self._encoder = self._make_encoder()
#
#
        num_conv_blocks = num_rn_blocks # TODO Was starting to make this ResNet-
        # based

        self._conv_out_dim = None
        self._conv_out_sz = None
        self._lin_size = None
        self._num_conv_blocks = num_conv_blocks
        self._Ws = None

        self._conv_out_dim = self._chip_size // (2**self._num_conv_blocks)
        self._conv_out_sz = self._base_nchans \
                * (2**(self._num_conv_blocks - 1))
        self._lin_size = self._conv_out_dim * self._conv_out_dim \
                * self._conv_out_sz
        self._make_Ws()

        self._encoder = conv_seq(self._Ws, is_transpose=False)
        self.bneck1 = nn.Linear(self._lin_size, self._bneck_size)
        self.bneck2 = nn.Linear(self._lin_size, self._bneck_size)
        self.fc = nn.Linear(self._bneck_size, self._lin_size)
        self._decoder = conv_seq(self._Ws, is_transpose=True)
        self.sigmoid = nn.Sigmoid()

    def _make_encoder(self):
        rnbs = OrderedDict()
        for b in range(self._num_rn_blocks):
            name = "rn_enc_%d" % (b+1)
#            rnb = ResBlock(
            

    def encode(self, x):
        if g_DEBUG: print("encode")
        x = self._encoder(x)
        if g_DEBUG: print(x.shape)
        x = x.view(-1, self._lin_size)
        bneck1 = self.bneck1(x)
        if g_DEBUG: print(bneck1.shape)
        bneck2 = self.bneck2(x)
        if g_DEBUG: print(bneck2.shape)
        return bneck1, bneck2
        # So returning mean (mu) and log of the variance (logvar), here

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable( std.data.new(std.size()).normal_() )
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        if g_DEBUG: print("decode")
        if g_DEBUG: print(z.shape)
        z = self.fc(z).view(-1, self._conv_out_sz, self._conv_out_dim,
                self._conv_out_dim)
        z = self._decoder(z)
        if g_DEBUG: print(z.shape)
        return self.sigmoid(z).view(-1, 1, self._chip_size, self._chip_size)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1, self._chip_size, 
            self._chip_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def _make_Ws(self):
        self._Ws = []
        num_back = self._num_conv_blocks // 2
        num_front = self._num_conv_blocks - num_back
        k_sizes = num_front * [5] + num_back * [3]
        for i in range(self._num_conv_blocks):
            if i==0:
                c_in = 1
            else:
                c_in = self._base_nchans * (2**(i-1))
            c_out = self._base_nchans * (2**i)
            ksz = k_sizes[i]
            W = torch.FloatTensor(c_out, c_in, ksz, ksz)
            self._Ws.append( Parameter(W) )



def _test_main(args):
    sz = args.chip_size
    if args.model == "CropNetAE":
        model = CropNetAE(chip_size=sz)
    elif args.model == "CropNetFCAE":
        model = CropNetFCAE()
    else:
        raise NotImplementedError()
    print("Using model %s" % (args.model))
    x = torch.FloatTensor(sz, sz).uniform_()
    print("x shape: %s" % repr(x.shape))
    yhat,mu,logvar = model(x)
    print("yhat shape: %s, mu shape %s, logvar shape %s" \
            % (yhat.shape, mu.shape, logvar.shape))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="CropNetAE",
            choices=["CropNetAE", "CropNetFCAE"])
    parser.add_argument("--chip-size", type=int, default=19)
    args = parser.parse_args()
    _test_main(args)
