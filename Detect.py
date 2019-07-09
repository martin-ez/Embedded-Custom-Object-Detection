import click
import os, json, requests, time, atexit
from datetime import datetime
from PIL import Image
from src.Model import Model
from src.VideoFeed import VideoFeed
from src.Helpers import encode_image, print_result, draw_boxes, convert_sample_rate, load_json

video_feed = None
sample_rate = None
api_url = None
model = None
output_path = None

@click.command()
@click.option('--image', help='Path to image to be detected')
@click.option('--out_folder', help='Path to save the output image')
def main(image, out_folder):
    print('---------------------------------')
    print('- EMBEDDED DETECTION PROCESS -')
    print('---------------------------------')
    print(' - LOADING CONFIGURATION FILE')
    global video_feed
    global sample_rate
    global api_url
    global model
    global output_path
    dirname = os.path.dirname(__file__)
    config = load_json(os.path.join(dirname, 'config.json'))
    classes = load_json(os.path.join(dirname, config['inference']['class_map'] + '.json'))
    sample_rate = convert_sample_rate(config['inference']['sample_rate'])
    api_url = config['inference']['api_url']
    model_config = {
    'conf_threshold': config['inference']['conf_threshold'],
    'model_path': os.path.join(dirname, 'models', config['inference']['model_name']),
    'classes': classes["classes"]
    }
    print(' | - Configuration set correctly')
    print(' | ')
    model = Model(model_config)
    if out_folder:
        output_path = os.path.expanduser(out_folder)
    if image:
        detect_local_image(os.path.expanduser(image))
    else:
        video_feed = VideoFeed(0)
        print(' | ')
        print(' - STARTING DETECTION LOOP')
        print(' | - Sample rate: ',sample_rate, ' seconds')
        print(' | ')

        detection_loop()

def detect_local_image(image_path):
    print(' - DETECTION ON LOCAL IMAGE')
    print(' | - Image path: ',image_path)
    print(' | ')
    image = Image.open(image_path)
    detect_image(image)

def detection_loop():
    while True:
        frame = video_feed.read()
        image = Image.fromarray(frame)
        detect_image(image)
        time.sleep(sample_rate)

def detect_image(image):
    print(' | - IMAGE DETECTION')
    now = datetime.now()
    timestamp = str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'-'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)
    print(' | | - Timestamp: ',timestamp)
    start_time = time.time()
    detection = model.detect(image)
    final_time = time.time() - start_time
    print_result(detection)
    print(' | | - Detection time: ', final_time, ' seconds')
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        draw_boxes(image, detection).save(os.path.join(output_path, timestamp + '.png'))
        print(' | | - Output image saved at ', os.path.join(output_path, timestamp + '.png'))
    else:
        files= {
            'raw_image': encode_image(image),
            'detection_image': encode_image(draw_boxes(image, detection)),
            'json': (None, json.dumps({'result': detection, 'timestamp': timestamp}), 'application/json')
        }
        r = requests.post(api_url, files=files)
        print(' | | - Detection results received by ', api_url)
    print(' | | ')

def cleanup():
    print(' - SHUTTING DOWN DETECTION PROCESS')
    if video_feed is not None:
        video_feed.release()

atexit.register(cleanup)

if __name__ == '__main__':
    main()
