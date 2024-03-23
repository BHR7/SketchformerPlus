import time
import torch
from options import TestOptions
from utils.data_util import load_data
from utils.strokeTool import draw_stroke,to_normal_strokes
from framework import SketchModel
from writer import Writer
import random
 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def test(opt,writer):
 
    # train_dataset
    test_loader = load_data(opt,datasetType='test', permutation=True,shuffle = opt.shuffle)

    model = SketchModel(opt)

    # training
    step = 0
    evaluate_count = 0
    for i, cpt_data in enumerate(test_loader):
        cpt_data = cpt_data.to(torch.float32)
        batch_id = random.randint(0,cpt_data.shape[0] - 1)
        data = cpt_data[batch_id]
        data = data.unsqueeze(0)

        output = model.predict(data)

        drawSketch = []
        drawSketch.append(data[0][1:].tolist())
        drawSketch.append(output[0][1:].tolist())

        draw_stroke(drawSketch,name='output/evaluate/',count=evaluate_count)
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