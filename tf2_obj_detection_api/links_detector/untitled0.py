import pathlib
import os
import tensorflow as tf
from tf2_obj_detection_api.models.research.object_detection.utils import (
    label_map_util)
from tf2_obj_detection_api.models.research.object_detection.utils import (
    config_util)
from tf2_obj_detection_api.models.research.object_detection.builders import (
    model_builder)


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

# Generates the detection function for specific model and specific model's checkpoint
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

import matplotlib.pyplot as plt
%matplotlib inline

# Creating a TensorFlow dataset of just one image.
inference_ds = tf.keras.preprocessing.image_dataset_from_directory(
directory='images',
image_size=(640, 640),
batch_size=1,
shuffle=False,
label_mode=None
  )
  # Numpy version of the dataset.
  inference_ds_numpy = list(inference_ds.as_numpy_iterator())

  # You may preview the images in dataset like this.
  plt.figure(figsize=(14, 14))
  for i, image in enumerate(inference_ds_numpy):
  plt.subplot(2, 2, i + 1)
  plt.imshow(image[0].astype("uint8"))
  plt.axis("off")
  plt.show()

a, b, c = inference_detect_fn('images/12-inference-01.jpg')


