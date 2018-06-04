import os
import sys
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.append(".")

PATH_TO_PRETRAIN_DET = 'faster_rcnn_resnet101_lowproposals_coco_2017_11_08.pb'
PATH_TO_PRETRAIN_CLS = 'pj_vehicle_inception_v4_freeze.pb'
PATH_TO_LABELS = 'labels.txt'
TEST_IMAGE_PATH = 'test.jpg'

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8), im_height, im_width

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
    """Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by
    applying resize_bilinear.

    If central_fraction is specified it would crop the central fraction of the
    input image.

    Args:
        image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
        height: integer
        width: integer
        central_fraction: Optional Float, fraction of the image to crop.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
        # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = saved_model_pb2.SavedModel()
    with tf.gfile.GFile(PATH_TO_PRETRAIN_DET, 'rb') as fid:
        serialized_graph = compat.as_bytes(fid.read())
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def.meta_graphs[0].graph_def, name='')

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        image = Image.open(TEST_IMAGE_PATH)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        (image_np, im_height, im_width) = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

classes = np.squeeze(classes).astype(np.int32)
boxes = np.squeeze(boxes)


car_index = np.where(classes == 3)[0][0]
bbox = boxes[car_index]
y1 = int(bbox[0] * im_height)
x1 = int(bbox[1] * im_width)
y2 = int(bbox[2] * im_height)
x2 = int(bbox[3] * im_width)
crop_height = y2-y1
crop_width = x2-x1

image_crop = tf.image.crop_to_bounding_box(
        image_np, 
        offset_height=y1, offset_width=x1, 
        target_height=crop_height, target_width=crop_width)

img_crop = Image.fromarray(tf.Session().run(image_crop), 'RGB')

with tf.name_scope('preprocess'):
    image_in = tf.placeholder(tf.uint8, shape=[None, None, 3], name='image_in')
    image_preprocessed = preprocess_for_eval(image_in, 299, 299)

img_preproc = tf.Session().run(image_preprocessed, feed_dict={image_in: img_crop})
img_np_expanded = np.expand_dims(img_preproc, axis=0)

with open(PATH_TO_PRETRAIN_CLS, 'rb') as f:
    cls_graph_def = tf.GraphDef()
    cls_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(cls_graph_def, name='')

sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('InceptionV4/Logits/Predictions:0')
predictions = sess.run(softmax_tensor, feed_dict={'input:0': img_np_expanded})
predictions = np.squeeze(predictions)
class_index = np.where(predictions == np.max(predictions))[0][0]
probs = np.max(predictions) * 100

with open(PATH_TO_LABELS) as f:
    category_index = {}
    for i in f.readlines():
        lns = i.strip('\n')
        for ln in lns.split(','):
            e = ln.strip().split(':')
            category_index[int(e[0])] = {'id': int(e[0]), 'name': e[1]}

im = cv2.imread(TEST_IMAGE_PATH, cv2.COLOR_BGR2RGB)
pil_im = Image.fromarray(im)
draw = ImageDraw.Draw(pil_im)
font = ImageFont.truetype('./simhei.ttf', 18, encoding='utf-8')

class_name = category_index[class_index]['name']
disp_str = class_name + ': ' + ('%s' % str(int(probs))) + '%'
len_disp_str = len(disp_str.encode('gb2312'))
draw.rectangle((x1, y1, x1 + len_disp_str*9, y1+18), 
               fill=(211, 211, 211), outline=None)
draw.text((x1, y1), disp_str, (255, 0, 255), font=font)
draw.line((x1, y1, x2, y1), (0, 255, 0), width=3)
draw.line((x2, y1, x2, y2), (0, 255, 0), width=3)
draw.line((x2, y2, x1, y2), (0, 255, 0), width=3)
draw.line((x1, y2, x1, y1), (0, 255, 0), width=3)

im = np.array(pil_im)
cv2.imwrite('out.jpg', im)