import numpy as np
import  matplotlib.pyplot as plt
import matplotlib.image as img
import torch
from scipy import interpolate
from torch.optim import lr_scheduler
import os
import random
from torch.autograd import Variable
from options import TrainOptions,TestOptions

opt = TrainOptions().parse()
DEVICE = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'warmup':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_ntokens(tgt_y):
    tgt_y = tgt_y.contiguous().view(-1, tgt_y.size(-1))
    
    n_tokens = 0
    for point in tgt_y: 
        if(point[0] != 0 and point[1] != 0):
            n_tokens += 1
    return n_tokens

def imgBinaryThreshold(img, threshold=128):
    """
    image binary threshold
    im: source image
    threshold:Threshold from 0 to 255
    Return gray image.
    """
    imgarray = np.array(img, dtype = np.uint8)
    rows = img.shape[0]
    cols = img.shape[1]
    for i in range(rows):
        for j in range(cols):
            gray = (imgarray[i, j, 0] * 0.299 + imgarray[i, j, 1] * 0.587 + imgarray[i, j, 2] * 0.114)
            if(gray < threshold):
                imgarray[i, j, :] = 0
            else:
                imgarray[i, j, :] = 255
    return imgarray.astype(np.uint8)

def create_padding_mask(seq):
    seq = seq.cpu().data.numpy()
    if len(seq.shape) < 3:  # tokenized version
        seq = torch.BoolTensor(np.equal(seq, 0))
    elif seq.shape[-1] > 1:  # continuous version (look at last bit)
        seq = torch.BoolTensor(seq[..., -1] != 1)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:,np.newaxis, np.newaxis,  :]  # (batch_size,1,1,seq_len)

def make_std_mask(tgt, pad):
    """Create a mask to hide padding and future words."""
    tgt_mask = (tgt[...,-1] != 1).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))
    return tgt_mask

def create_look_ahead_mask(size):
    # create an lower tri and invert it to get an upper trianguler with no diag
    #mask = 1 - tf.linalg.band_part(torch.ones((size, size)), -1, 0)
    subsequent_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 0

def create_masks(inp,tar):

    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = dec_target_padding_mask & Variable(look_ahead_mask.type_as(dec_target_padding_mask.data))

    return enc_padding_mask, combined_mask, dec_padding_mask


def create_masks_test(src,tgt):

    #src_mask = (src != 0).unsqueeze(-2)
    src_mask = torch.FloatTensor((src[..., -1] == 1).float())
    if tgt is not None:
        tgt = tgt.to(DEVICE)
        
        trg_mask = make_std_mask(tgt, 0)

        #ntokens = (tgt_y != 0).data.sum()

    return src_mask,trg_mask



def delete_point(points):

    ori_index = np.arange(len(points))
    del_index = np.random.choice(len(points), len(points) - 190, replace=False)

    index = np.delete(ori_index, del_index)

    return points[index]

def subsequent_mask(size):
    """Mask out subsequent positions."""
    
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')


    return torch.from_numpy(subsequent_mask) == 0

if __name__ == "__main__":


    import numpy as np