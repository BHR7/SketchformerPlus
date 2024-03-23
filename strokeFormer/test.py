import time
import torch
from options import TestOptions
from utils.data_util import load_data
from utils.strokeTool import segment2point, draw_strokes
from framework import SketchModel
from writer import Writer
import random
import numpy as np
from segmentFormer.options import TestOptions as segmentOption
from segmentFormer.framework import SketchModel as segmentModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
def test(opt,writer):
    
    opt_segmentTest = segmentOption().parse()

    segmentDecoder = segmentModel(opt_segmentTest)
    # test_dataset
    test_loader = load_data(opt,datasetType='test', permutation=True,shuffle = False)

    model = SketchModel(opt)

    # testing
    step = 0
    evaluate_count = 0
    for i, cpt_data in enumerate(test_loader):


        output = model.predict(cpt_data.to(torch.float32))
                # print(f'output.shape:{output.shape}')
    
        stroke = segment2point(reconStroke=output.cpu(),segmentDecoder=segmentDecoder,device=device)  
        gts = segment2point(reconStroke=cpt_data.cpu(),segmentDecoder=segmentDecoder,device=device)  

        for s,gt in zip(stroke, gts):
            
            draw_strokes(s,svg_filename=f'output/test/testSeq/pred/recon_{evaluate_count}.svg',png_filename=f'output/test/testSeq/pred/recon_{evaluate_count}.png')
            evaluate_count += 1
        


        step += 1


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