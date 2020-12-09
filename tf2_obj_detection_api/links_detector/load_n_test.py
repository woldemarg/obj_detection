import pathlib
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tf2_obj_detection_api.models.research.object_detection.utils import (
    label_map_util)
from tf2_obj_detection_api.models.research.object_detection.utils import (
    config_util)
from tf2_obj_detection_api.models.research.object_detection.builders import (
    model_builder)
from tf2_obj_detection_api.models.research.object_detection.utils import (
    visualization_utils)

# %%

MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
TF_MODELS_BASE_PATH = 'http://download.tensorflow.org/models/object_detection/tf2/'
CACHE_FOLDER = 'tf2_obj_detection_api/links_detector'


# %%

def download_tf_model(base_path, model_date, model_name, cache_folder):
    model_url = (base_path +
                 model_date +
                 '/' +
                 model_name +
                 '.tar.gz')
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=model_url,
        untar=True,
        cache_dir=pathlib.Path(cache_folder).absolute()
    )
    return model_dir


# %%

# Start the model download.
model_dir = download_tf_model(TF_MODELS_BASE_PATH,
                              MODEL_DATE,
                              MODEL_NAME,
                              CACHE_FOLDER)


# %%

def load_coco_labels():
    label_map_path = os.path.join(
        r'tf2_obj_detection_api\models\research\object_detection\data',
        'mscoco_complete_label_map.pbtxt'
    )
    label_map = label_map_util.load_labelmap(label_map_path)

    # Class ID to Class Name mapping.
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)

    # Class Name to Class ID mapping.
    label_map_dict = (label_map_util
                      .get_label_map_dict(label_map,
                                          use_display_name=True))

    return category_index, label_map_dict


# %%

# Load COCO labels.
coco_category_index, coco_label_map_dict = load_coco_labels()


# %%

# Generates the detection function for specific model
# and specific model's checkpoint
def detection_fn_from_checkpoint(config_path, checkpoint_path):
    # Build the model.
    pipeline_config = config_util.get_configs_from_pipeline_file(config_path)
    model_config = pipeline_config['model']
    model = model_builder.build(
        model_config=model_config,
        is_training=False)

    # Restore checkpoints.
    ckpt = tf.compat.v2.train.Checkpoint(model=model)
    ckpt.restore(checkpoint_path).expect_partial()

    # This is a function that will do the detection.
    @tf.function
    def detect_fn(image):
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


# %%

inference_detect_fn = detection_fn_from_checkpoint(
    config_path=os.path.join(r'tf2_obj_detection_api\links_detector',
                             'datasets',
                             MODEL_NAME,
                             'pipeline.config'),
    checkpoint_path=os.path.join(r'tf2_obj_detection_api\links_detector',
                                 'datasets',
                                 MODEL_NAME,
                                 'checkpoint',

                                 'ckpt-0'))

# %%

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


# %%

image_np = load_image_into_numpy_array('images/boats.jpg')

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),
                                    dtype=tf.float32)

detections, predictions_dict, shapes = inference_detect_fn(
      input_tensor
  )

# %%

boxes = detections['detection_boxes'].numpy()
scores = detections['detection_scores'].numpy()
classes = detections['detection_classes'].numpy()
num_detections = detections['num_detections'].numpy()[0]

print('boxes.shape: ', boxes.shape)
print('scores.shape: ', scores.shape)
print('classes.shape: ', classes.shape)
print('num_detections:', num_detections)

# %%

print('First 5 boxes:')
print(boxes[0,:5])

print('First 5 scores:')
print(scores[0,:5])

print('First 5 classes:')
print(classes[0,:5])

class_names = [coco_category_index[idx + 1]['name'] for idx in classes[0]]
print('First 5 class names:')
print(class_names[:5])


# %%

# Visualizes the bounding boxes on top of the image.
def visualize_detections(image_np, detections, category_index):
    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        ((detections['detection_classes'][0].numpy() + label_id_offset)
         .astype(int)),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.4,
        agnostic_mode=False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np_with_detections)
    plt.show()


# %%

# Visualizing the detections.
visualize_detections(
    image_np=image_np,
    detections=detections,
    category_index=coco_category_index,
    )
