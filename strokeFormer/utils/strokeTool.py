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


def to_normal_strokes(big_stroke):  #batch_size x seq_len x 130
    """Convert from stroke-4 format (from sketch-rnn paper) back to stroke-3.
    
    input: batch_size x seq_len x 4
    return: (batch_size x seq_len) x 3
    """
    #result = np.zeros((big_stroke.shape[0],big_stroke.shape[1], 3))
    
    result = np.zeros((big_stroke.shape[0] * big_stroke.shape[1], big_stroke.shape[2]))

    l = 0 
    total_len = 0

    for stroke in big_stroke:
        r = 0
        for i in range(len(stroke)):
            if np.argmax(stroke[i, :]) == (big_stroke.shape[2] - 1):
                r = i
                break
        if r == 0:
            r = len(stroke)

        result[l:l + r - 1,:] = stroke[1:r, :]
        result[l+r-1,129] = 1

        total_len = total_len + r

        l = l + r

    return result[:total_len]


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

def draw_stroke(stroke,name,count):

    for s in stroke:
        s = np.cumsum(s,axis=0)
        plt.scatter(s[:-1,0],-s[:-1,1],s=1)
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()          
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                canvas.tostring_rgb())

    pil_image.save(name + str(count)+'.jpg',"JPEG")
    plt.close("all")

def draw_strokes(data, factor=0.01, svg_filename = "",png_filename = ""):
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
    
    the_color = "black"
    stroke_width = 2.
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    drawing = svg2rlg(svg_filename)
    renderPM.drawToFile(drawing, png_filename, fmt="PNG")


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


def resample(stroke,max_len):

    if(len(stroke) < max_len - 1):
        stroke = np.row_stack((np.zeros(3),stroke))
        return np.array(stroke)
    else:
        sample_list = [i for i in range(len(stroke))] # [0, 1, 2, 3]
        sample_list = random.sample(sample_list, max_len-2) # [1, 2]
        stroke = stroke[sample_list,:] 
        stroke = np.row_stack((np.zeros(3),stroke))
        return np.array(stroke)
    
def getStrokeSegment(stroke, absoluteStroke, theta):
    result = []
    temp = []
    length = 0

    if(len(stroke) == 1):
        return np.array(stroke)

    for i in range((len(stroke)-1)):
        temp.append(stroke[i])
        length += np.sqrt(np.sum(np.power(absoluteStroke[i+1] - absoluteStroke[i], 2)[:2]))
        if length >= theta:
            length = 0
            result.append(temp)
            temp = []
    
    temp.append(stroke[len(stroke)-1])
    if(len(temp)): result.append(temp)

    return np.array(result)

def segment2point(reconStroke, segmentDecoder, device):

    #reconStroke: batch_size x seq_len x 130
    #return: batch_size x 80 x 3

    strokeList = np.zeros((reconStroke.shape[0],80,3))

    count = 0 

    for stroke in reconStroke:
        seq_len = 0

        for segmentId in range(1, stroke.shape[0]):
            if np.argmax(stroke[segmentId,128:]) == 1: #end token
                seq_len = segmentId
                break
            
        if(seq_len == 0):
            seq_len = len(stroke)
        
        segment = segmentDecoder.predict_from_embedding(stroke[1:seq_len,:stroke.shape[1] - 2].to(device))
        strokeList[count][:len(segment)] = segment[:]

        count += 1

    return strokeList