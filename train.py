import time
import torch
from options import TrainOptions
from utils.data_util import load_data
#from framework import SketchModel
from framework import SketchModel
from writer import Writer
import random
from utils.util import setup_seed
from utils.strokeTool import draw_stroke
import numpy as np
 
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
def train(opt,writer):
    setup_seed(1234)
    # train_dataset
    train_loader = load_data(opt,datasetType='train', permutation=True,shuffle = opt.shuffle)
    #valid_loader = load_data(opt,datasetType='valid', permutation=True,shuffle = opt.shuffle)

    model = SketchModel(opt)
    #model.print_detail()

    #training
    step = 0
    min_loss = 1000

    for epoch in range(opt.epochs):
        #pd = tqdm(train_loader)
        i_step = 0
        mloss = 1000

        for i, (cpt_data,label) in enumerate(train_loader):
            cpt_data = cpt_data.to(torch.float32)

            inp = tar = cpt_data
            model.step(inp,tar,label)

            if(model.loss < mloss):
                mloss = model.loss
            if i % opt.plot_freq == 0:
                writer.plot_train_loss(model.loss, step)
            if i % opt.print_freq == 0:
                writer.print_train_loss(epoch, i, model.loss)
    
            i_step += 1
            step += 1
        if(mloss < min_loss):
            min_loss = mloss
            model.save_network('bestloss')
        model.update_learning_rate()
            
        
    writer.close()
    

def main():
    torch.manual_seed(777)
    opt_train = TrainOptions().parse()

    writer = Writer(opt_train)

    train(opt_train,writer)
 
 
if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total train time is ', time_end - time_start)