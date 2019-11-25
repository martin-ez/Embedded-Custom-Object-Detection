from io import BytesIO
from PIL import ImageDraw, ImageFont
from re import match, sub
from sys import exit
from random import randint
from os import path
from ast import literal_eval
import numpy as np
import json, base64
import cv2

def load_json(file_path):
    with open(file_path) as json_file:
        decoded = json.load(json_file)
    return decoded

def encode_image(image):
    bytes = BytesIO()
    image = image.convert("RGB")
    image.save(bytes, format='JPEG')
    img_str = base64.b64encode(bytes.getvalue()).decode('utf-8')
    return "data:image/jpeg;base64,"+str(img_str)

def print_result(detection):
    print(' | | - Detected ', str(len(detection)), ' objects')
    for element in detection:
        print(' | | |-- ' + element['physical_element_code'] + ' - H: ' + str(element['height']) + ' W: ' + str(element['width']))


def class_color_code(classes):
    colorcode = {}
    for cls in classes:
        colorcode[cls] = get_random_color()
    return colorcode

def get_random_color():
    return tuple([randint(0,200), randint(0,200), randint(0,200)])

def draw_boxes(image, detection, colorcode):
    font_path = path.join(path.dirname(__file__), 'font', 'Montserrat.ttf')
    thickness = 5
    font = ImageFont.truetype(font=font_path, size=24)
    drawImage = image.copy()
    draw = ImageDraw.Draw(drawImage)
    for element in detection:
        cl = element['physical_element_code']
        cl_color = colorcode[cl]
        b = element['values']
        label = '{} {}%'.format(cl, int(element['score']*100))
        label_size = draw.textsize(label, font)
        left, right, top, bottom = b
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=cl_color)
        draw.rectangle([left, bottom - label_size[1], left + label_size[0], bottom], fill=cl_color)
        draw.text([left, bottom - label_size[1]], label, fill=(255, 255, 255), font=font)
    del draw
    return drawImage

def convert_sample_rate(sample_rate):
    if match('^[0-9]+ms$', sample_rate) is not None:
        return int(sample_rate.split('ms')[0])/1000
    elif match('^[0-9]+s$', sample_rate) is not None:
        return int(sample_rate.split('s')[0])
    else:
        exit(' | - ERROR: Invalid sample rate format')

def replace(text, replacements):
    replace_text = text
    for (regEx, replc) in replacements:
        replace_text = sub(regEx, replc, replace_text)
    return replace_text

def pil2opencv(pil_image):
    open_cv_image = np.array(pil_image)
    return open_cv_image[:, :, ::-1].copy()

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation = inter)

def show_image(img):
    img = pil2opencv(img)
    cv2.imshow('Detection', image_resize(img, height=800))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
