from email.policy import default
import os
import time
import torch
import json
import argparse
import numpy as np
import random

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def initialize(self):
        # data params
        self.parser.add_argument('--limit', type=int, default=1000)
        self.parser.add_argument('--augment_stroke_prob', type=float, default=0.1)
        self.parser.add_argument('--random_scale_factor', type=float, default=0.1)
        self.parser.add_argument('--augment', type=bool, default=True)
        self.parser.add_argument('--src_vocab', type=int, default=130)
        self.parser.add_argument('--tgt_vocab', type=int, default=130)
        self.parser.add_argument('--dataset-name', type=str, required=False, default='quick_draw')
        self.parser.add_argument('--dataset', type=str, required=False, default='sketch')
        self.parser.add_argument('--perm-type', type=str, default='-')
        self.parser.add_argument('--num-workers', type=int, default=0, help='numworkers')
        self.parser.add_argument('--gpu-ids', type=str, default='1', 
                    help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--net-name', type=str, default='segmentTransfromer')
        #transformer
        self.parser.add_argument('--segment-len', type=int, default=20)
        self.parser.add_argument('--stroke-len', type=int, default=20)
        self.parser.add_argument('--d-model', type=int, default=256)
        self.parser.add_argument('--nhead', type=int, default=8)
        self.parser.add_argument('--low_dim', type=int, default=256)
        self.parser.add_argument('--pos_dim', type=int, default=10003)
        self.parser.add_argument('--dim-feedforward', type=int, default=1024)
        self.parser.add_argument('--dropout', type=float, default=0.1)
        self.parser.add_argument('--activation', type=str, default="relu")
        self.parser.add_argument('--num-layers', type=int, default=4)
        self.parser.add_argument('--num-encoder-layers', type=int, default=4)
        self.parser.add_argument('--num-decoder-layers', type=int, default=4)
        

        self.parser.add_argument('--checkpoints-dir', type=str, default='param/encoder_decoder/')

        self.parser.add_argument('--which-epoch', type=str, default='bestloss', 
                    help='which epoch to load? set to latest to use latest cached model')
        
    def parse(self, params=None):
        if not self.initialized:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        #self.opt.timestamp =  time.strftime("%b%d_%H")
        self.opt.timestamp =  time.strftime("%b%d")
        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self):
        
        self.parser = argparse.ArgumentParser(description='strokeGae')
        BaseOptions.initialize(self)

        self.parser.add_argument('--seed', default=1234, type=int, help='random seed')
        self.parser.add_argument('--batch-size', type=int, default=64, 
                                 help='intout batch size')
        self.parser.add_argument('--epochs',type=int,default=400)
        self.parser.add_argument('--learning_rate',type=float,default=0.0001)
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--plot-freq', type=int, default=1, 
                                 help='frequency of ploting training loss')
        self.parser.add_argument('--print-freq', type=int, default=100, 
                                 help='frequency of showing training loss on console')
        self.parser.add_argument('--is-train', type=bool, default=True)
        self.parser.add_argument('--shuffle', type=bool, default=True)
        self.parser.add_argument('--lr-policy', type=str, default='warmup', 
                                 help='learning rate policy: lambda|step|plateau|warmup')
        self.parser.add_argument('--lr-decay-iters', type=int, default=100, 
                                 help='multiply by a gamma every lr_decay_iters iterations')

        args = self.parser.parse_args()

 
class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--timestamp', type=str, default='-', 
                help='the timestep of the model')
        self.parser.add_argument('--print-freq', type=int, default=2, 
                help='frequency of showing training results on console')
        self.parser.add_argument("--permutation", action='store_true',
                                 help='if data permutation')
        self.parser.add_argument('--batch-size', type=int, default=64, help='intout batch size')
        self.parser.add_argument('--encoder-decoder', type=str, default='param/encoder_decoder/encoder_decoder_net_bestloss.pkl')
        self.parser.add_argument('--encoder', type=str, default='param/encoder/encoder_net_bestloss.pkl')
        self.parser.add_argument('--shuffle', type=bool, default=True)
        self.parser.add_argument('--plot_every',type=int,default=200)
        self.parser.add_argument('--epochs',type=int,default=1)
        self.parser.add_argument('--is-train', type=bool, default=False)
        self.is_train = False



if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import ParameterSampler

    _opt = TrainOptions().parse()
    with open(_opt.randomlist, 'r') as f:
        param_grid = json.load(f)
    rng = np.random.RandomState(0)
    param_list = list(ParameterSampler(param_grid, n_iter=5, random_state=rng))

    for params in param_list:
        opt = TrainOptions().parse(params)
        print(params)
