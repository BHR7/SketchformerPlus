import os.path as osp
import numpy as np
from utils.strokeTool import get_bounds,augment_strokes,split_sketch,resample,convert_to_absolute,getStrokeSegment
from torch.utils.data import Dataset
import numpy as np
import matplotlib.image as img
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class StrokeDataset(Dataset):
    def __init__(self, opt, root,dataset_name, split='train', permutation=False):
        self.dataset_name = dataset_name
        self.split = split
        self.max_len = opt.seq_len
        self.dataset = opt.dataset
        #self.json_dir = osp.join(root, '{}_{}.ndjson'.format(self.dataset_name, self.split))
        self.npz_dir = osp.join(root, '{}_{}.npz'.format(self.dataset, self.split))
        self.theta_dir = osp.join(root, 'theta.npy')
        
        self.limit = opt.limit
        self.augment = opt.augment
        self.augment_stroke_prob = opt.augment_stroke_prob
        self.random_scale_factor = opt.random_scale_factor
        self.processed_data = self._process()

    def __getitem__(self, index):
        return self.processed_data[index]
    
    def __len__(self):
        return len(self.processed_data)

    def _cap_pad_and_convert_stroke(self, stroke):
        desired_length = self.max_len
        stro_len = len(stroke)

        converted_sketch = np.zeros((desired_length, 4), dtype=float)
        converted_sketch[:stro_len, 0:2] = stroke[:, 0:2]
        converted_sketch[:stro_len, 2] = stroke[:, 2]
        #converted_sketch[:stro_len, 2] = 1 - stroke[:, 2]
        converted_sketch[stro_len:, 3] = 1
        converted_sketch[-1:, 3] = 1

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
        count = 0

        print(f'classNum:{len(raw_data)}')
        max_segments = 0

        for i in range(len(raw_data)):
            
            theta = thetaList[i] 

            sketchs = raw_data[i]

            end_sketchs = len(sketchs)
            
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
                    
                    segments = getStrokeSegment(stroke, absoluteStroke, theta)

                    max_segments = max(len(segments), max_segments)
                    
                    for segment in segments:
                        
                        segment = resample(np.array(segment),max_len=self.max_len).round(8)
                        
                        segment[:,2] = 1

                        segment = self._cap_pad_and_convert_stroke(segment).round(8)

                        processed_data.append(segment)
                        
                        count += 1

        print(f'strokeNum:{len(processed_data)}')
        print(f'max_segments:{max_segments}')
        return np.array(processed_data)


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
    count = 0
    for i,data in enumerate(dataloader):
        print(data.shape)
        print(data[0])
        stroke = convert_to_absolute(data[0].tolist())
        print(stroke)
        plt.clf()
        #plt.scatter(cpt_data[1:cpt_length+1,0],cpt_data[1:cpt_length+1,1])
        #plt.plot(stroke[:,0],stroke[:,1])
        plt.scatter(stroke[:,0],stroke[:,1],s=1)
        plt.savefig(f'output/test_cpt/{i}.png')
        

    
    ###draw sketch
    '''
    draw_sketch = sketch[:,:2]
    draw_sketch = np.cumsum(draw_sketch,axis=0)
    draw_sketch2 = sketch
    draw_sketch2[:,:2] = draw_sketch[:,:2]
    draw_sketch2[:,2] = sketch[:,2]

    draw_sketch = split_sketch(draw_sketch2)
    '''
    '''
    name = "output/test_cpt/"
    for s in draw_sketch:
        plt.plot(s[1:,0],-s[1:,1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()          
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                canvas.tostring_rgb())
    pil_image.save(name + str(count)+'.jpg',"JPEG")
    plt.close("all")
    '''