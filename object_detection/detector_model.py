import os

import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util


class Detector:

    def __init__(self, num_classes):
        model_name = 'object_detection/checkpoint'

        cwd_path = os.getcwd()

        path_to_checkpoint = os.path.join(cwd_path, model_name, 'frozen_inference_graph.pb')
        path_to_labels = os.path.join(cwd_path, 'object_detection/checkpoint', 'labelmap.pbtxt')

        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=num_classes,
            use_display_name=True
        )
        self.category_index = label_map_util.create_category_index(categories)

        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_checkpoint, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.graph)

        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    def detection(self, image):
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded}
        )
        return boxes, scores, classes, num
