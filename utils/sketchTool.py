import numpy as np
from scipy import interpolate
import random
from PIL import Image
import PIL
import matplotlib.image as img
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import svgwrite
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import cycle
import os.path as osp

from utils.util import mkdir


def slerp(p0, p1, t):
    """Spherical interpolation."""
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    if so < 1e-6:  # p0 = p1
        return p0
    else:
        return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

def lerp(p0, p1, t):
    """Linear interpolation."""
    return (1.0 - t) * p0 + t * p1

def lines_to_strokes(lines, omit_first_point=True):
    """
    Convert polyline format to stroke-3 format.
    lines: list of strokes, each stroke has format Nx2
    """
    strokes = []
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :] if omit_first_point else strokes

def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)

def split_sketch(sketch): 
    strokes = np.split(sketch, np.where(sketch[:,2]>0)[0]+1)
    return strokes

def getSketchLen(raw_data):  

    min_Len = 10000
    max_len = -1
    totalNum = 0

    for i in range(len(raw_data)):
        sketchs = raw_data[i]

        end_sketchs = len(sketchs) // 2
        sketchs = sketchs[:end_sketchs]

        totalNum += len(sketchs)

        for sketch in sketchs:
            strokeNum = len(split_sketch(sketch))

            if(strokeNum < min_Len):
                min_Len = strokeNum
            if(strokeNum > max_len):
                max_len = strokeNum
            

    return min_Len,max_len,totalNum

def to_normal_strokes(big_stroke):
    """Convert from stroke-4 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if np.argmax(big_stroke[i, :]) == 3:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    #result[:, 2] = big_stroke[0:l, 3]
    result[-1, 2] = 1
    return result


def strokes_to_lines(strokes, scale=1.0, start_from_origin=False):
    """
    convert strokes3 to polyline format ie. absolute x-y coordinates
    note: the sketch can be negative
    :param strokes: stroke3, Nx3
    :param scale: scale factor applied on stroke3
    :param start_from_origin: sketch starts from [0,0] if True
    :return: list of strokes, each stroke has format Nx2
    """
    x = 0
    y = 0
    lines = []
    line = [[0, 0]] if start_from_origin else []
    for i in range(len(strokes)):
        x_, y_ = strokes[i, :2] * scale
        x += x_
        y += y_
        line.append([x, y])
        if strokes[i, 2] == 1:
            line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
            lines.append(line_array)
            line = []
    if lines == []:
        line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
        lines.append(line_array)
    return lines

def draw_sketch(reconSketch,sketch,width,current_sektch,svg_filename,png_filename):

    #sketch_list = build_interlaced_grid_list2(reconSketch[np.newaxis,:,:], sketch[np.newaxis,:,:], width = width,current_sketch=current_sektch)
    sketch_list = build_interlaced_grid_list2(reconSketch, sketch, width = width,current_sketch=current_sektch)
    sketch_grid = make_grid_svg(sketch_list)

    draw_strokes(sketch_grid,svg_filename=svg_filename,png_filename=png_filename)

def get_absolute_bounds(data, factor=1):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    return min_x, max_x, min_y, max_y


def get_bounds(data, factor=1):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)

def convert_to_absolute(sketch):
    absolute_sketch = np.zeros_like(sketch)
    absolute_sketch[0] = sketch[0]
    for i, (prev, new, orig) in enumerate(zip(absolute_sketch, absolute_sketch[1:], sketch[1:])):
        new[:2] = prev[:2] + orig[:2]
        new[2:] = orig[2:]
    return absolute_sketch

def getLen(sketch):

    length = 0
    for i in range(len(sketch) - 1):
        if(sketch[i][2] == sketch[i + 1][2] == 1):
            break
        length += 1

    return length

def to_relative(stroke, factor=1):
    relative_stroke = np.zeros_like(stroke)
    relative_stroke[0] = stroke[0]
    relative_stroke[0, :2] = stroke[0, :2] * factor
    for i, (prev_orig, new, orig) in enumerate(zip(stroke, relative_stroke[1:], stroke[1:])):
        new[:2] = (orig[:2] - prev_orig[:2]) * factor
        new[2:] = orig[2:]
    return relative_stroke

def list_to_relative(stroke):
    relative_stroke = []
    relative_stroke.append(to_relative(stroke))
    return relative_stroke

def strokeResample(stroke,max_len):

    if(len(stroke) < max_len - 1):
        # point = np.row_stack((np.zeros(2),point))
        # point = np.column_stack((point,np.zeros(len(point))))
        # return point
        num_point = len(stroke)
        stroke = np.row_stack((np.zeros(3),stroke))
        #stroke = np.column_stack((stroke,np.ones(num_point + 1)))  #[x,y,0]

        return np.array(stroke)
    else:

        sample_list = [i for i in range(len(stroke))] # [0, 1, 2, 3]
        sample_list = random.sample(sample_list, max_len-2) # [1, 2]
        
        stroke = stroke[sample_list,:] 

        num_point = max_len - 2

        stroke = np.row_stack((np.zeros(3),stroke))
        #stroke = np.column_stack((stroke.tolist(),np.ones(num_point + 1)))  #[x,y,0]
        return np.array(stroke)

def sketchResample(sketch):
    #input   [batch,embedding
    strokeNum = len(sketch)
    print(sketch.shape)
    sketch = np.row_stack((np.zeros(sketch.shape[1]),sketch)) 
    sketch = np.column_stack((sketch.tolist(),np.ones((strokeNum + 1,2))))  

    print(sketch.shape)
    return np.array(sketch)

def build_interlaced_grid_list2(targets, preds,width,current_sketch):  
    grid_list = []
    for i in range(0, width, 2):
        for j in range(width):
            grid_list.append([targets[current_sketch], [i, j]])
            try:
                grid_list.append([preds[current_sketch], [i + 1, j]])
            except:
                pass
            current_sketch += 1
    return grid_list

def make_grid_svg(s_list, grid_space=2.25, grid_space_x=2.5):
    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max + x_min) * 0.5
        return x_start - center_loc, x_end
    x_pos = 0.0
    y_pos = 0.0
    result = []
    for sample in s_list:
        sketch = sample[0]
        if len(sketch) == 0:
            continue
        sketch[0, -1] = 1
        grid_loc = sample[1]
        grid_y = grid_loc[0] * grid_space + grid_space * 0.5
        grid_x = grid_loc[1] * grid_space_x + grid_space_x * 0.5
        start_loc, delta_pos = get_start_and_end(sketch)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x + loc_x
        new_y_pos = grid_y + loc_y
        result.append([new_x_pos - x_pos, new_y_pos - y_pos, 0])

        result += sketch.tolist()
        if result[-1][2] == 1:
            result[-2][2] = 0
        else:
            result[-1][2] = 1
        x_pos = new_x_pos + delta_pos[0]
        y_pos = new_y_pos + delta_pos[1]
    return np.array(result)

def draw_strokes(data, factor=0.01, svg_filename = "",png_filename = ""):
    '''draw one sketch'''
    #svg_filenamei = osp.join(svg_filename,f'{777}.svg')
    min_x, max_x, min_y, max_y = get_bounds(data,factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    
    the_color = "#2c3e50"
    stroke_width = 2.
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    #drawing = svg2rlg(svg_filename)
    #renderPM.drawToFile(drawing, png_filename, fmt="PNG")
    #print("drawing....")

def draw_sketch_stroke(data, factor=0.01, svg_filename = "",png_filename = ""):
    '''get diff color for stroke'''
    min_x, max_x, min_y, max_y = get_bounds(data,factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y

    the_color = ["blue", "red", "green", "blue", "purple", "pink", "orange"]

    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    count = 0
    stroke_width = 2.
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        #lift_pen = data[i, 2]
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "

        if (lift_pen == 1):
            count += 1

            svg_filenamei = osp.join(svg_filename,f'{count}.svg')
            dwgi = svgwrite.Drawing(svg_filenamei, size=dims)
            dwgi.add(dwgi.rect(insert=(0, 0), size=dims, fill='white'))
            dwgi.add(dwgi.path(p).stroke(the_color[count % 7], stroke_width).fill("none"))
            dwgi.save()

            p = "M%s,%s " % (abs_x, abs_y)
            
    
    # the_color = "black"
    # stroke_width = 2.
    # dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    # dwg.save()
    # drawing = svg2rlg(svg_filename)
    # renderPM.drawToFile(drawing, png_filename, fmt="PNG")
    print("drawing....")

def sketch2point(reconSketch, strokeDecoder,segmentDecoder, device):

    #reconSketch: batch_size x seq_len x 258，one batch one sketch
    #return: batch_size x 80 x 3

    sketchList = np.zeros((reconSketch.shape[0],500,3))

    count = 0 

    for sketch in reconSketch:
        #seq_len x 258
        sketch_len = 0 

        for strokeId in range(1, sketch.shape[0]):
            if np.argmax(sketch[strokeId, 256:]) == 1: #end token
                sketch_len = strokeId
                break

        if(sketch_len == 0):
            sketch_len = len(sketch)
        
        reconSketch = strokeDecoder.predict_from_embedding(sketch[1:sketch_len,:sketch.shape[1] - 2].to(device)).cpu()
        
        strokeList = np.zeros((reconSketch.shape[0]*150,3))

        l = 0
        total_len = 0
        for stroke in reconSketch:

            seq_len = 0

            for segmentId in range(1, stroke.shape[0]):
                if np.argmax(stroke[segmentId,128:]) == 1: #end token
                    seq_len = segmentId
                    break
                
            if(seq_len == 0):
                seq_len = len(stroke)
            
            segment = segmentDecoder.predict_from_embedding(stroke[1:seq_len,:stroke.shape[1] - 2].to(device))


            r = len(segment)

            strokeList[l : l + r] = segment[:]
            strokeList[l + r, 2] = 1

            l = l + r + 1
            total_len = total_len + r + 1
            # l = l + r 
            # total_len = total_len + r
        
        sketchList[count][:total_len] = strokeList[:total_len]
        sketchList[count][total_len][2] = 1  #represent one sketch end
        count += 1

    return sketchList
