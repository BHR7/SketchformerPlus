import os.path as osp
import numpy as np
from utils.strokeTool import get_bounds,augment_strokes,split_sketch,resample,convert_to_absolute,getStrokeSegment
from torch.utils.data import Dataset
import numpy as np
from segmentFormer.options import TestOptions as segmentOption
from segmentFormer.framework import SketchModel as segmentModel
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class StrokeDataset(Dataset):
    def __init__(self, opt, root,dataset_name, split='train', permutation=False):
        self.dataset_name = dataset_name
        self.split = split
        self.segment_len = opt.segment_len
        self.stroke_len = opt.stroke_len
        self.dataset = opt.dataset
        #self.json_dir = osp.join(root, '{}_{}.ndjson'.format(self.dataset_name, self.split))
        self.npz_dir = osp.join(root, '{}_{}.npz'.format(self.dataset, self.split))
        self.theta_dir = osp.join(root, 'theta.npy')
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.limit = opt.limit
        self.augment = opt.augment
        self.augment_stroke_prob = opt.augment_stroke_prob
        self.random_scale_factor = opt.random_scale_factor

        self.opt_segmentTest = segmentOption().parse()

        self.segmentEncoder = segmentModel(self.opt_segmentTest)
        self.processed_data = self._process()


    def __getitem__(self, index):
        return self.processed_data[index]
    
    def __len__(self):
        return len(self.processed_data)

    def _cap_pad_and_convert_segment(self, segment):
        desired_length = self.segment_len
        stro_len = len(segment)

        converted_segment = np.zeros((desired_length, 4), dtype=float)
        converted_segment[:stro_len, 0:2] = segment[:, 0:2]
        converted_segment[:stro_len, 2] = segment[:, 2]
        #converted_sketch[:stro_len, 2] = 1 - stroke[:, 2]
        converted_segment[stro_len:, 3] = 1
        converted_segment[-1:, 3] = 1

        return converted_segment
    
    def _cap_pad_and_convert_stroke(self,stroke):
        desired_length = self.stroke_len
        stroke_len = len(stroke)
        stroke_len = stroke.shape[0]
        embeddingLen = stroke.shape[1] - 2
        converted_stroke = np.zeros((desired_length, stroke.shape[1]), dtype=float)
        converted_stroke[:stroke_len, 0:embeddingLen] = stroke[:, 0:embeddingLen]
        converted_stroke[:stroke_len, -2] = stroke[:, -2]
        #converted_sketch[:stro_len, 2] = 1 - stroke[:, 2]
        converted_stroke[stroke_len:, -1] = 1
        converted_stroke[-1:, -1] = 1

        #print(f'converted_stroke.shape:{converted_stroke.shape}')
        return converted_stroke
    
    def random_scale(self, data):
        """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
        x_scale_factor = (
            np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (
            np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result
    
    def _augment_sketch(self, sketch, set_type='train'):
        if self.augment_stroke_prob > 0 and set_type == 'train':
            data_raw = self.random_scale(sketch)
            data = np.copy(data_raw)
            data = augment_strokes(data, self.augment_stroke_prob)
            return data
        else:
            return sketch

    def _process(self, perm_arg=None):
        print(f'Loading{self.npz_dir}_dataset')
        if perm_arg is not None:
            print('Processing with augment param {} ...'.format(perm_arg))
        else:
            print('Processing without augment param ...')

        raw_data = []
        # with open(self.json_dir, 'r') as f:
        #     for line in f:
        #         raw_data.append(json.loads(line)["drawing"])
        raw_data = np.load(self.npz_dir, encoding='latin1', allow_pickle=True)['sketch']
        thetaList = np.load(self.theta_dir)

        processed_data = []

        gtData = []
        count = 0

        print(f'classNum:{len(raw_data)}')
        max_segments = 0

        for i in range(len(raw_data)):
            
            theta = thetaList[i] 

            sketchs = raw_data[i]

            #end_sketchs = len(sketchs) // 10

            end_sketchs = 10
            
            sketchs = sketchs[:end_sketchs]

            for sketch in sketchs:

                # removes large gaps from the data
                sketch = np.minimum(sketch, self.limit)
                sketch = np.maximum(sketch, -self.limit)
                sketch = np.array(sketch, dtype=np.float32)

                # augment if required
                sketch = self._augment_sketch(sketch) if self.augment else sketch

                # get bounds of sketch and use them to normalise
                min_x, max_x, min_y, max_y = get_bounds(sketch)
                max_dim = max([max_x - min_x, max_y - min_y, 1])
                sketch[:, :2] /= max_dim

                absoluteSketch = convert_to_absolute(sketch)

                absoluteSketch = split_sketch(absoluteSketch)

                sketch = split_sketch(sketch)

                for stroke,absoluteStroke in zip(sketch,absoluteSketch):
                        
                    if (len(stroke) < 1):
                        continue
                    
                    segments = getStrokeSegment(stroke, absoluteStroke, theta)

                    
                    strokeSample = [] 

                    for segment in segments:
                        
                        segment = resample(np.array(segment),max_len=self.segment_len).round(8)
                        
                        segment[:,2] = 1

                        segment = self._cap_pad_and_convert_segment(segment).round(8)

                        strokeSample.append(segment)
                        #processed_data.append(segment)
                        #count += 1

                    strokeSample = torch.tensor(strokeSample)

                    #print(f'strokeSample.shape:{strokeSample.shape}')
                    embedding,_ = self.segmentEncoder.encode_from_seq(torch.tensor(strokeSample).to(self.device))

                    strokeEmb =np.array(embedding.detach().cpu().numpy())
                    strokeNum = len(strokeEmb)

                    #print(f'strokeEmb.shape:{strokeEmb.shape}')  # [strokeNum, embedding]
                    strokeEmb = np.row_stack((np.zeros(strokeEmb.shape[1]),strokeEmb))   
                    strokeEmb = np.column_stack((strokeEmb.tolist(),np.zeros((strokeNum + 1,2))))  

                    strokeEmb[:,-2] = 1
                    if(strokeEmb.shape[0] > self.stroke_len):
                        strokeEmb = strokeEmb[:self.stroke_len]
                    
                    strokeEmb = self._cap_pad_and_convert_stroke(strokeEmb)



                    processed_data.append(strokeEmb)

                    count += 1

        return np.array(processed_data)


# ---- methods for batch collation ----

if __name__ == "__main__":
    import os
    import torch
    from utils.data_util import load_data
    from strokeFormer.utils.strokeTool import convert_to_absolute,segment2point
    from options import TestOptions,TrainOptions
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("hello")
    #opt = TestOptions().parse()
    opt = TrainOptions().parse()
    dataloader = load_data(opt,datasetType='train', permutation=True)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    opt_segmentTest = segmentOption().parse()

    segmentDecoder = segmentModel(opt_segmentTest)
    count = 0
    for i,data in enumerate(dataloader):
        print(data.shape)
        print(data[0])

        reconStroke = data

        stroke = segment2point(reconStroke=reconStroke,segmentDecoder=segmentDecoder,device=device)

        for s in stroke:
            draw_stroke = s[:,:2]
            draw_stroke = np.cumsum(draw_stroke,axis=0)
            plt.clf()
            #plt.scatter(cpt_data[1:cpt_length+1,0],cpt_data[1:cpt_length+1,1])
            #plt.plot(stroke[:,0],stroke[:,1])
            plt.scatter(draw_stroke[:,0],draw_stroke[:,1],s=1)
            plt.savefig(f'output/test_cpt/{i}.png')

