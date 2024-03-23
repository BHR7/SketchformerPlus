import os.path as osp
import numpy as np
from utils.strokeTool import get_bounds,augment_strokes,split_sketch,resample,convert_to_absolute,getStrokeSegment
from torch.utils.data import Dataset
import numpy as np
from strokeFormer.segmentFormer.options import TestOptions as segmentOption
from strokeFormer.segmentFormer.framework import SketchModel as segmentModel
from strokeFormer.options import TestOptions as strokeOption
from strokeFormer.framework import SketchModel as strokeModel
import joblib
import torch
import matplotlib
matplotlib.use('Agg')
import random
import matplotlib.pyplot as plt

class StrokeDataset(Dataset):
    def __init__(self, opt, root,dataset_name, split='train', permutation=False):
        self.dataset_name = dataset_name
        self.split = split
        self.segment_len = opt.segment_len
        self.stroke_len = opt.stroke_len
        self.sketch_len = opt.sketch_len

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

        self.pt_dirData = osp.join(root, '{}_data_{}.pkl'.format(self.dataset, self.split))
        self.pt_dirLabel = osp.join(root, '{}_label_{}.pkl'.format(self.dataset, self.split))
        self.pt_gtData = osp.join(root, '{}_gt_{}.pkl'.format(self.dataset, self.split))



        if osp.exists(self.pt_dirData):
            self.label_data = joblib.load(self.pt_dirLabel)
            self.processed_data = joblib.load(self.pt_dirData)

            self.gt_data = joblib.load(self.pt_gtData)

        else:
            self.processed_data, self.label_data,self.gt_data = self._process()

    def __getitem__(self, index):
        return self.processed_data[index],self.label_data[index],self.gt_data[index]
    
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
    
    def _cap_pad_and_convert_sketch(self,sketch):
        desired_length = self.sketch_len
        sketch_len = len(sketch)
        sketch_len = sketch.shape[0]
        embeddingLen = sketch.shape[1] - 2
        converted_sketch = np.zeros((desired_length, sketch.shape[1]), dtype=float)
        converted_sketch[:sketch_len, 0:embeddingLen] = sketch[:, 0:embeddingLen]
        converted_sketch[:sketch_len, -2] = sketch[:, -2]
        #converted_sketch[:stro_len, 2] = 1 - stroke[:, 2]
        converted_sketch[sketch_len:, -1] = 1
        converted_sketch[-1:, -1] = 1


        #print(f'converted_sketch.shape:{converted_sketch.shape}')
        return converted_sketch
    
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

        self.opt_segmentTest = segmentOption().parse()
        self.segmentEncoder = segmentModel(self.opt_segmentTest)

        self.opt_strokeTest = strokeOption().parse()
        self.strokeEncoder = strokeModel(self.opt_strokeTest)
        print(f'Loading {self.npz_dir} _dataset')
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
        label_data = []
        gt_data = []

        count = 0
        print(f'classNum:{len(raw_data)}')
        max_stroke = 0

        for i in range(len(raw_data)):
            theta = thetaList[i] 

            sketchs = raw_data[i]

            end_sketchs = len(sketchs)

            sketchs = sketchs[:end_sketchs]

            for sketch in sketchs:
                label_data.append(i)
                
                # removes large gaps from the data
                sketch = np.minimum(sketch, self.limit)
                sketch = np.maximum(sketch, -self.limit)
                sketch = np.array(sketch, dtype=np.float32)
                
                min_x, max_x, min_y, max_y = get_bounds(sketch)
                max_dim = max([max_x - min_x, max_y - min_y, 1])
                sketch[:, :2] /= max_dim

                if(len(sketch) > 256):
                    sketch = sketch[:256]

                sk2 = np.zeros((256,3))
                sk2[:len(sketch)] = sketch[:]

                absoluteSketch = convert_to_absolute(sketch)
                absSketch = absoluteSketch
                absoluteSketch = split_sketch(absoluteSketch)

                sketch = split_sketch(sketch)[:-1]
                sketchSample = []
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

                    strokeSample = torch.tensor(strokeSample)
                    embedding,_ = self.segmentEncoder.encode_from_seq(torch.tensor(strokeSample).to(self.device))

                    strokeEmb =np.array(embedding.detach().cpu().numpy())
                    strokeNum = len(strokeEmb)

                    strokeEmb = np.row_stack((np.zeros(strokeEmb.shape[1]),strokeEmb))   
                    strokeEmb = np.column_stack((strokeEmb.tolist(),np.zeros((strokeNum + 1,2))))  

                    strokeEmb[:,-2] = 1
                    if(strokeEmb.shape[0] > self.stroke_len):
                        strokeEmb = strokeEmb[:self.stroke_len]
                    
                    strokeEmb = self._cap_pad_and_convert_stroke(strokeEmb)

                    sketchSample.append(strokeEmb)


                sketchSample = torch.tensor(sketchSample)
                
                if(len(sketchSample) < 1):
                    continue
                embedding,_ = self.strokeEncoder.encode_from_seq(torch.tensor(sketchSample).to(self.device))
                

                sketchEmb = np.array(embedding.detach().cpu().numpy())
                sketchNum = len(sketchEmb)

                max_stroke = max(max_stroke, sketchNum)  

                #print(f'strokeEmb.shape:{strokeEmb.shape}')  # [strokeNum, embedding]
                sketchEmb = np.row_stack((np.zeros(sketchEmb.shape[1]),sketchEmb))   
                sketchEmb = np.column_stack((sketchEmb.tolist(),np.zeros((sketchNum + 1,2))))  

                sketchEmb[:,-2] = 1
                if(sketchEmb.shape[0] > self.sketch_len):
                    sketchEmb = sketchEmb[:self.sketch_len]
                

                sketchEmb = self._cap_pad_and_convert_sketch(sketchEmb)

                gt_data.append(sk2)
                processed_data.append(sketchEmb)

 
        joblib.dump(processed_data, self.pt_dirData)
        joblib.dump(label_data, self.pt_dirLabel)
        joblib.dump(gt_data, self.pt_gtData)

        print(f'{len(processed_data)}-------{len(label_data)}-------{len(gt_data)}')
        return np.array(processed_data),np.array(label_data),np.array(gt_data)


# ---- methods for batch collation ----

if __name__ == "__main__":
    import os
    import torch
    from utils.data_util import load_data
    from utils.strokeTool import convert_to_absolute
    from options import TestOptions,TrainOptions
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    #opt = TestOptions().parse()
    opt = TrainOptions().parse()
    dataloader = load_data(opt,datasetType='train', permutation=True)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    count = 0
    for i,(data,label) in enumerate(dataloader):
        print(data.shape)

        break