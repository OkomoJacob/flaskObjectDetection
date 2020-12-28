import os #to acces files from your pc
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
# from werkzeug import secure_filename #For security of our app
import numpy as np
import six.moves.urllib as urllib
import sys #to access system files
import tensorflow as tf #NN Classifier
from collections import defaultdict
from io import StringIO
from PIL import Image
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_utils
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_LABELS = os.path.join('data','mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# detections
detection_graph = tf.Graph() #initialize a detection graph
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.Gfile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name = '')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_class = NUM_CLASSES, use_display_name = True) 
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
# routing and uploading
@app.route('/')
def index():
    return render_template('index.html')

def upload():
    file = request.files(['file'])
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename = filename))
# Once we upload the img, we need to route to the uploads for detects
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range (1, 2)]
    IMAGE_SIZE = (12, 8)
    
    with detection_graph.as_default():
        with tf.Session(graph = detection_graph) as sess:
            for image_path in TEST_IMAGE_PATH:
                image = image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis = 0)
                image_tesnsor = detection_graph.get_tensor_by_name('image_tensors: 0')
                boxes = detection_graph.get_tensor_by_name('detection_box: 0')
                scores = detection_graph.get_tensor_by_name('detection_scores: 0')
                classes = detection_graph.get_tensor_by_name('detection_classes: 0')
                num_detections = detection_graph.get_tensor_by_name('num_detections: 0')
                (boxex, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict = {image_tensor:image_np_expanded})
                vis_utils.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalize_coordinates = True, line_thickness = 8)
                im = Image.fromarray(array_np)
                im.save('uploads/' + filename)
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    if __name__ == '__main__':
        app.run(debug = True, host = '0.0.0.0', port = 5000)

