from io import BytesIO
from PIL import ImageDraw, ImageFont
from re import match
from sys import exit
from random import randint
from os import path
import json

def load_json(file_path):
    with open(file_path) as json_file:
        decoded = json.load(json_file)
    return decoded

def encode_image(image):
    bytes = BytesIO()
    image.save(bytes, format='JPEG')
    bytes.seek(0)
    return bytes

def print_result(detection):
    total_objects = 0
    objects_per_class=[]
    for cl, boxes in detection.items():
        n_objects = len(boxes)
        total_objects += n_objects
        objects_per_class.append(' | | |-- ' + str(n_objects) + ' ' + cl)
    print(' | | - Detected ', str(total_objects), ' objects')
    for ls in objects_per_class:
        print(ls)

def draw_boxes(image, detection):
    font_path = path.join(path.dirname(__file__), 'font', 'Montserrat.ttf')
    thickness = 5
    font = ImageFont.truetype(font=font_path, size=24)
    draw = ImageDraw.Draw(image)
    for cl, boxes in detection.items():
        cl_color = get_random_color()
        for b in boxes:
            label = '{} {}%'.format(cl, int(b['score']*100))
            label_size = draw.textsize(label, font)
            left, right, top, bottom = b['box']
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=cl_color)
            draw.rectangle([left, bottom - label_size[1], left + label_size[0], bottom], fill=cl_color)
            draw.text([left, bottom - label_size[1]], label, fill=(255, 255, 255), font=font)
    del draw
    return image

def get_random_color():
    return tuple([randint(0,200), randint(0,200), randint(0,200)])

def convert_sample_rate(sample_rate):
    if match('^[0-9]+ms$', sample_rate) is not None:
        return int(sample_rate.split('ms')[0])/1000
    elif match('^[0-9]+s$', sample_rate) is not None:
        return int(sample_rate.split('s')[0])
    else:
        exit(' | - ERROR: Invalid sample rate format')
