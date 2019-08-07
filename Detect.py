import os, json, requests, time, atexit, cv2
from absl import flags, app
from datetime import datetime
from PIL import Image
from src.Model import Model
from src.VideoFeed import VideoFeed
from src.Helpers import encode_image, print_result, draw_boxes, convert_sample_rate, load_json, show_image

FLAGS = flags.FLAGS
flags.DEFINE_string('image', None, 'Path to the image that will be detected. When omitted, a continuous detection using a web camera will begin.')
flags.DEFINE_string('output', None, 'Path to a folder where the resulting images will be saved.')
flags.DEFINE_boolean('sendresults', False, 'Detection process will send the results of the detection to the Rest API found on the config.json file.')
flags.DEFINE_boolean('dontshow', False, 'Detection images will not be shown on the screen.')

video_feed = None
sample_rate = None
api_url = None
model = None
output_path = None

def main(_):
    print('---------------------------------')
    print('      - DETECTION PROCESS -')
    print('---------------------------------')
    print(' - LOADING CONFIGURATION FILE')
    global video_feed
    global sample_rate
    global api_url
    global model
    global output_path
    dirname = os.path.dirname(__file__)
    config = load_json(os.path.join(dirname, 'config.json'))['inference']
    classes = load_json(os.path.join(dirname, 'data', config['class_map'] + '.json'))['classes']
    sample_rate = convert_sample_rate(config['sample_rate'])
    api_url = config['api_url']
    model_config = {
    'conf_threshold': config['conf_threshold'],
    'model_path': os.path.join(dirname, 'models', config['model_name']),
    'classes': classes
    }
    print(' | - Configuration set correctly')
    print(' | ')
    model = Model(model_config)
    if FLAGS.output:
        output_path = os.path.expanduser(FLAGS.output)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    if FLAGS.image:
        detect_local_image(os.path.expanduser(FLAGS.image))
    else:
        video_feed = VideoFeed(0, display=not FLAGS.dontshow)
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

def display_image(image):
    if FLAGS.image is not None:
        show_image(image)
    else:
        video_feed.display(image)

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
    output_img = draw_boxes(image, detection)
    if output_path:
        output_img.save(os.path.join(output_path, timestamp + '.png'))
        print(' | | - Output image saved at ', os.path.join(output_path, timestamp + '.png'))
    if not FLAGS.dontshow:
        display_image(output_img)
    if FLAGS.sendresults:
        files= {
            'raw_image': encode_image(image),
            'detection_image': encode_image(output_img),
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
    app.run(main)
