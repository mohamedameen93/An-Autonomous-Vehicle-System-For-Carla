import tensorflow as tf
from styx_msgs.msg import TrafficLight


PROTOBUG_GRAPH_FILE = 'light_classification/retrained_SSD/frozen_inference_graph.pb'


class TLClassifier(object):
    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PROTOBUG_GRAPH_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=graph)

        with self.sess.as_default():
            self.image_ = graph.get_tensor_by_name('image_tensor:0')
            self.boxes_ = graph.get_tensor_by_name('detection_boxes:0')
            self.scores_ = graph.get_tensor_by_name('detection_scores:0')
            self.classes_ = graph.get_tensor_by_name('detection_classes:0')

        self.category_index = {
            1: TrafficLight.GREEN,
            2: TrafficLight.RED,
            3: TrafficLight.YELLOW,
            4: TrafficLight.UNKNOWN,
        }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.sess.as_default():
            tensors_ = [self.boxes_, self.scores_, self.classes_]
            feed_dict = {self.image_: [image]}
            boxes, scores, classes = self.sess.run(tensors_, feed_dict)

        most_probable_class = int(classes[0][0])
        return self.category_index[most_probable_class]
