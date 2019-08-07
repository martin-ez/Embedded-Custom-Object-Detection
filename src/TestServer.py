import json, io, os
from PIL import Image
import numpy as np
from flask import Flask, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def decode_image(img):
    buffer = io.BytesIO()
    img.save(buffer)
    return Image.open(buffer)

def decode_json(data):
    payload = json.loads(data)
    return payload['result'], payload['timestamp']

def process_detection(image, detection, time):
    print('---------------------------------')
    print_result(detection, time)
    out_path = os.path.join(os.path.dirname(__file__), 'server_out')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_path = os.path.join(out_path, time+'.png')
    image.save(img_path)
    print('  Result image saved at '+img_path)

def print_result(detection, time):
    total_objects = 0
    objects_per_class=[]
    for cls in detection:
        n_objects = len(cls['instances'])
        total_objects += n_objects
        objects_per_class.append('  - '+str(n_objects)+' '+cls['class'])
    print('+ Detection '+str(time))
    print('  Detected '+str(total_objects)+' objects:')
    for ls in objects_per_class:
        print(ls)

@app.route('/detection', methods=['POST'])
def detect():
    r = request
    image = decode_image(r.files['raw_image'])
    res_image = decode_image(r.files['detection_image'])
    detection, time = decode_json(r.form['json'])
    process_detection(res_image, detection, time)
    return Response(response={'detections': 'success'}, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run()
