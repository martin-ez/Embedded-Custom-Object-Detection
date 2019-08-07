import os, json, sys
import tensorflow as tf
import numpy as np
from PIL import Image

class Model:

    def __init__(self, config):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
        print(' - LOADING MODEL')
        print(' | - Setting up model configuration and classes')
        self.classes = config['classes']
        self.conf_threshold = config["conf_threshold"]
        print(' | - Loading weights')
        PATH_TO_FROZEN_GRAPH = os.path.join(config['model_path'], 'frozen_inference_graph.pb')
        sys.path.append('..')
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.graph)

        detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.graph.get_tensor_by_name('num_detections:0')
        self.tensor_dict = [detection_boxes, detection_scores, detection_classes, num_detections]
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        print(' | - Warming up model')
        self.detect(Image.new('RGB', (640, 640), (128,128,128)))
        print(' | - Model loaded successfully')
        print(' | ')

    def _preprocess(self, image, size=(640, 640)):
        resized = image.resize(size, Image.BICUBIC)
        image_np = np.array(resized.getdata()).reshape((size[0], size[0], 3)).astype(np.uint8)
        return np.expand_dims(image_np, axis=0)

    def detect(self, image):
        image_data = self._preprocess(image)
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: image_data})
        return self._postprocess(output_dict, image.size)

    def _postprocess(self, output_dict, image_shape):
        # all outputs are float32 numpy arrays, so convert types as appropriate
        boxes = self.transform_boxes(output_dict[0][0], image_shape)
        scores = output_dict[1][0]
        classes = output_dict[2][0].astype(np.int64)
        num_detections = int(output_dict[3][0])

        class_bag = {}
        detection = []
        for i in range(num_detections):
            box, clss, score = boxes[i], classes[i], scores[i]
            if score > self.conf_threshold:
                class_name = 'NA'
                if clss >= 0 and clss < len(self.classes):
                    class_name = self.classes[clss]
                if class_name not in class_bag:
                    class_bag[class_name] = []
                class_bag[class_name].append({
                'box': box.tolist(),
                'score': float(score)
                })
        for cls in class_bag:
            detection.append({
                'class': cls,
                'instances': class_bag[cls]
            })
        return detection

    def transform_boxes(self, boxes, image_shape):
        im_width, im_height = image_shape

        for i in range(len(boxes)):
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            boxes[i] = (left, right, top, bottom)

        return boxes
