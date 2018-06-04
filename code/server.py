##################
# import modules #
##################
import os
import sys
import time
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# opencv-python
import cv2
# copy file function
from shutil import copy2
from flask import Flask, request, redirect, send_from_directory, url_for
import uuid

################
# define FLAGS #
################
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('class_model_path', 
                           'pj_vehicle_inception_v4_freeze.pb', """model for classification""")
tf.app.flags.DEFINE_string('detection_model_path', 
                           'faster_rcnn_resnet101_lowproposals_coco_2017_11_08.pb', """model for detection""")
tf.app.flags.DEFINE_string('label_file', 'labels.txt', """classification labels""")
tf.app.flags.DEFINE_string('upload_folder', './uploads', """folder for save upload images""")
tf.app.flags.DEFINE_string('output_folder', './outputs', """folder for save upload images""")
tf.app.flags.DEFINE_integer('port', '5001', """server with port; if no port, use default port 80""")
tf.app.flags.DEFINE_boolean('debug', False, """if debug""")

# folder to save uploaded images
UPLOAD_FOLDER = FLAGS.upload_folder
# allowed image file extension name
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG'])

app = Flask(__name__)
app._static_folder = FLAGS.output_folder

def allowed_files(filename):
    # check if file is allowed
    return '.' in filename and \
            filename.rsplit('.', 1)[-1] in ALLOWED_EXTENSIONS

def rename_filename(old_filename):
    # replace the primary filename with a uuid
    basename = os.path.basename(old_filename)
    (name, ext) = os.path.splitext(basename)
    new_filename = str(uuid.uuid1()) + ext
    return new_filename

def create_category_index(label_file):
    # create the classindex-classname dictionary from the label file
    with open(label_file) as f:
        category_index = {}
        for i in f.readlines():
            lns = i.strip('\n')
            for ln in lns.split(','):
                index_and_name = ln.strip().split(':')
                category_index[int(index_and_name[0])] = {
                        'id': int(index_and_name[0]), 'name': index_and_name[1]}
    return category_index

def load_image_into_numpy_array(image):
    # load a image into numpy array
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8), im_height, im_width

def preprocess_for_class_infer(image, height, width,
                        central_fraction=0.875, scope=None):
    """
    Prepare one image for classification inference.

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

def inference(filename, od_model_path, cls_model_path):
    # load the frozen object detection model and make inference on the image
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = saved_model_pb2.SavedModel()
        with tf.gfile.GFile(od_model_path, 'rb') as fid:
            serialized_graph = compat.as_bytes(fid.read())
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def.meta_graphs[0].graph_def, name='')
        
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

            image = Image.open(filename)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            (image_np, im_height, im_width) = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # squeeze classes, boxes, scores
            classes = np.squeeze(classes).astype(np.int32)
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
    
    # find the car box index with the max level of confidence, get its confidence level value
    car_index = np.where(classes == 3)[0][0]
    score = scores[car_index]
    if score > 0.2:
        # if confidence > 0.2, make classification inference on the bounding box part
        bbox = boxes[car_index]
        y1 = int(bbox[0] * im_height)
        x1 = int(bbox[1] * im_width)
        y2 = int(bbox[2] * im_height)
        x2 = int(bbox[3] * im_width)
        # bounding box height and width
        crop_height = y2 - y1
        crop_width = x2 - x1
        # crop the bounding box part
        with tf.name_scope('image_crop_by_bbox'):
            image_crop = tf.image.crop_to_bounding_box(
                    image_np, 
                    offset_height=y1, offset_width=x1, 
                    target_height=crop_height, target_width=crop_width)
        img_crop = Image.fromarray(tf.Session().run(image_crop), 'RGB')

        # preprocess for classification inference
        with tf.name_scope('preprocess'):
            image_in = tf.placeholder(tf.uint8, shape=[None, None, 3], name='image_in')
            image_preprocessed = preprocess_for_class_infer(image_in, 299, 299)
        img_preproc = tf.Session().run(image_preprocessed, feed_dict={image_in: img_crop})
        img_np_expanded = np.expand_dims(img_preproc, axis=0)

        # load the frozen classification model
        class_graph = tf.Graph()
        with class_graph.as_default():
            with open(cls_model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')
            # make inference
            with tf.Session(graph=class_graph) as sess:
                softmax_tensor = sess.graph.get_tensor_by_name('InceptionV4/Logits/Predictions:0')
                predictions = sess.run(softmax_tensor, feed_dict={'input:0': img_np_expanded})
                predictions = np.squeeze(predictions)
                class_index = np.where(predictions == np.max(predictions))[0][0]
                prob = np.max(predictions) * 100
                category_index = create_category_index(label_file=FLAGS.label_file)
                class_name = category_index[class_index]['name']
        
        # read in image file and initialize drawing
        im = cv2.imread(filename, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(im)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype('./simhei.ttf', 16, encoding='utf-8')
        
        # combine the display string to "write"
        class_name = category_index[class_index]['name']
        disp_str = class_name + ': ' + ('%s' % str(int(prob))) + '%'
        # define the text background width, draw text background
        len_disp_str = len(disp_str.encode('gb2312'))
        draw.rectangle((x1, y1, x1 + len_disp_str*8, y1+16), 
                    fill=(211, 211, 211), outline=None)
        # "write" the display string
        draw.text((x1, y1), disp_str, (255, 0, 255), font=font)
        # draw the bounding box
        draw.line((x1, y1, x2, y1), (0, 255, 0), width=3)
        draw.line((x2, y1, x2, y2), (0, 255, 0), width=3)
        draw.line((x2, y2, x1, y2), (0, 255, 0), width=3)
        draw.line((x1, y2, x1, y1), (0, 255, 0), width=3)

        # save drawing
        im = np.array(pil_im)
        cv2.imwrite(os.path.join(FLAGS.output_folder, os.path.basename(filename)), im)

    else:
        # confidence level <= 0.2, means "there's no car"
        prob = 0.0
        class_name = 'There\'s no car!'
    
    if score > 0.2:
        # if there's any car, show the result image on the webpage
        new_url = '/static/%s' % os.path.basename(filename)
        image_tag = '<img src="%s"></img><p>'
        new_tag = image_tag % new_url
    else:
        # if there's no car, show the primary image
        copy2(filename, os.path.join(FLAGS.output_folder, os.path.basename(filename)))
        new_url = '/static/%s' % os.path.basename(filename)
        image_tag = '<img src="%s"></img><p>'
        new_tag = image_tag % new_url
    
    # print classification result and score
    format_string = '%s (score: %.1f%%)' % (class_name, prob)
    ret_string = new_tag  + format_string + '<BR>' 
    return ret_string


@app.route("/", methods=['GET', 'POST'])
def root():
    result = """
        <!doctype html>
        <title>实战车辆检测及型号识别</title>
        <h1>请 feed 一张图片</h1>
        <form action="" method=post enctype=multipart/form-data>
        <p><input type=file name=file value='选择图片'>
            <input type=submit value='上传'>
        </form>
        <p>%s</p>
        """ % "<br>"
    # if post request, upload the image, and make inference
    if request.method == 'POST':
        file = request.files['file']
        old_file_name = file.filename
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            type_name = 'N/A'
            print('file saved to %s' % file_path)
            start_time = time.time()
            out_html = inference(file_path, 
                                 od_model_path=FLAGS.detection_model_path, 
                                 cls_model_path=FLAGS.class_model_path)
            # inference cost time
            duration = time.time() - start_time
            print('duration:[%.0fms]' % (duration*1000))
            return result + out_html 
    return result

if __name__ == "__main__":
    print('listening on port %d' % FLAGS.port)
    sess = tf.Session()
    app.sess = sess
    app.run(host='127.0.0.1', port=FLAGS.port, debug=FLAGS.debug, threaded=True)
