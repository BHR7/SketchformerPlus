import time
import torch
from options import TestOptions
from utils.data_util import load_data
from utils.sketchTool import sketch2point,draw_sketch
from framework import SketchModel
from writer import Writer
from strokeFormer.segmentFormer.options import TestOptions as segmentOption
from strokeFormer.segmentFormer.framework import SketchModel as segmentModel
from strokeFormer.options import TestOptions as strokeOption
from strokeFormer.framework import SketchModel as strokeModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
 
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
def test(opt,writer):
    

    opt_segmentTest = segmentOption().parse()
    segmentEncoder = segmentModel(opt_segmentTest)

    opt_strokeTest = strokeOption().parse()
    strokeEncoder = strokeModel(opt_strokeTest)
    # train_dataset
    test_loader = load_data(opt,datasetType='test', permutation=True,shuffle = False)

    model = SketchModel(opt)

    # testing

    np.set_printoptions(threshold=np.inf)
    for i, (cpt_data,label,gt_data) in enumerate(test_loader):

        output,_ = model.predict(cpt_data.to(torch.float32),label)
        
        reconSketchs = sketch2point(reconSketch=output.cpu(),
                               strokeDecoder=strokeEncoder,
                               segmentDecoder=segmentEncoder,device=device
                               )
    
        draw_sketch(reconSketchs, gt_data, svg_filename=f'output/testRecon/sample{i}.svg',png_filename='')

    writer.close()
    

def main():
    torch.manual_seed(777)
    opt_test = TestOptions().parse()
    #opt_test = TestOptions().parse()

    writer = Writer(opt_test)
    #write = opt_train.write
    # if write:
    #     timenow = str(datetime.now())[0:-10]
    #     timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    #     writepath = 'runs/{}'.format(timenow)
    #     if os.path.exists(writepath): shutil.rmtree(writepath)
    #     writer = SummaryWriter(log_dir=writepath)
    
    test(opt_test,writer)
 
 
if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total train time is ', time_end - time_start)