import os, glob
from sys import exit
import xml.etree.ElementTree as ET
from src.Helpers import load_json
import tensorflow as tf
import src.dataset_util as dataset_util

dirname = os.path.dirname(__file__)
classes = None
total_objects = {
    'train': 0,
    'test': 0
}
class_instances = {
    'train': {},
    'test': {}
}
tfrecord_paths = []

def load_classes():
    global classes
    global class_instances
    print(' | - Loading classes file')
    classes = load_json(os.path.join(dirname, 'classes.json'))['classes']
    for folder in ['train', 'test']:
        for x in range(1, len(classes)):
            class_instances[folder][classes[x]] = 0
    print(' | ')


def read_folder(folder):
    folder_path = os.path.join(dirname, 'data', folder)
    print(' - CONVERTING IMAGES IN:', folder_path)
    files = []
    print(' | - Reading xml files')
    for xml_file in glob.glob(os.path.join(folder_path, '*.xml')):
        filename, size, objects = read_xml(xml_file, folder)
        files.append({
            'filename': filename,
            'size': size,
            'objects': objects
        })
    return files

def read_xml(xml_file, folder):
    global total_objects
    global class_instances
    objects = []
    root = ET.parse(xml_file).getroot()
    filename = root.find('filename').text
    size = (int(root.find('size')[0].text), int(root.find('size')[1].text))
    for obj in root.findall('object'):
        class_name = obj[0].text
        try:
            class_no = classes.index(class_name)
        except:
            exit('- ERROR: Unrecognize class ' + class_name + ' in ' + xml_file)
        instance = {
            'class_name': class_name,
            'class_no': class_no,
            'box': (int(obj[4][0].text), int(obj[4][1].text), int(obj[4][2].text), int(obj[4][3].text))
        }
        total_objects[folder] += 1
        class_instances[folder][class_name] += 1
        objects.append(instance)
    return filename, size, objects

def write_tfrecord(labeled_imgs, folder):
    global tfrecord_paths
    print(' | - Writing TFRecord')
    out_path = os.path.join(dirname, 'data', folder+'.record')
    writer = tf.io.TFRecordWriter(out_path)
    images_path = os.path.join(dirname, 'data', folder)
    for img_data in labeled_imgs:
        tf_example = generate_tf_example(img_data, images_path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    tfrecord_paths.append(out_path)
    print(' | | - TFRecord for', folder, 'set completed')
    print(' | ')

def generate_tf_example(img_data, path):
    with tf.io.gfile.GFile(os.path.join(path, img_data['filename']), 'rb') as fid:
        encoded_jpg = fid.read()
    width, height = img_data['size']
    filename = img_data['filename'].encode('utf8')
    image_format = img_data['filename'].split('.')[1].encode('utf8')
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes_text = []
    classes = []

    for obj in img_data['objects']:
        classes_text.append(obj['class_name'].encode('utf8'))
        classes.append(obj['class_no'])
        xmins.append(obj['box'][0] / float(width))
        ymins.append(obj['box'][1] / float(height))
        xmaxs.append(obj['box'][2] / float(width))
        ymaxs.append(obj['box'][3] / float(height))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def print_dataset_analysis():
    print('---------------------------------')
    print(' - DATASET ANALYSIS')
    print('---------------------------------')
    for folder in ['train', 'test']:
        print(' | -', folder.upper() + 'ING SET')
        print(' | | - Total class instances:', total_objects[folder])
        for cl, count in class_instances[folder].items():
            print(' | | |--', cl+':', count, '-', '{0:.1%}'.format(count/float(total_objects[folder])))

        print(' | |')
    print(' | - TFRecords files')
    for record_path in tfrecord_paths:
        print(' | | -', record_path)
    print(' | ')

def main(_):
    print('---------------------------------')
    print('- DATASET PREPARATION -')
    print('---------------------------------')
    load_classes()
    for folder in ['train', 'test']:
        labeled_imgs = read_folder(folder)
        write_tfrecord(labeled_imgs, folder)
    print_dataset_analysis()

if __name__ == '__main__':
    tf.app.run()
