import time
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tbl_detection.models.research.object_detection.utils import (
    label_map_util)
from tbl_detection.models.research.object_detection.utils import (
    visualization_utils)
from tbl_detection.models.research.object_detection.data_decoders.tf_example_decoder import (
    TfExampleDecoder)
from imutils.object_detection import non_max_suppression
from tbl_detection import config

# %matplotlib inline

# %%

PATH_TO_SAVED_MODEL = config.SAVED_MODEL


# %%

def detection_function_from_saved_model(saved_model_path):
    print('Loading saved model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(saved_model_path)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('[INFO] Done! Took {} seconds'.format(math.ceil(elapsed_time)))

    return detect_fn


# %%

exported_detect_fn = detection_function_from_saved_model(
    PATH_TO_SAVED_MODEL)

# %%

category_index = label_map_util.create_category_index_from_labelmap(
      config.LABEL_MAP,
      use_display_name=True)

# %%

def tensors_from_tfrecord(
    tfrecords_filename,
    tfrecords_num,
    dtype=tf.float32
):
    decoder = TfExampleDecoder()
    raw_dataset = tf.data.TFRecordDataset(tfrecords_filename)
    images = []

    for raw_record in raw_dataset.take(tfrecords_num):
        example = decoder.decode(raw_record)
        image = example['image']
        image = tf.cast(image, dtype=dtype)
        images.append(image)

    return images


def test_detection(tfrecords_filename, tfrecords_num, detect_fn):
    image_tensors = tensors_from_tfrecord(
        tfrecords_filename,
        tfrecords_num,
        dtype=tf.uint8
    )

    for image_tensor in image_tensors:
        image_np = image_tensor.numpy()

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = tf.expand_dims(image_tensor, 0)

        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))

        detections = {key: value[0, :num_detections].numpy() for key, value
                      in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = (detections['detection_classes']
                                           .astype(np.int64))

        image_np_with_detections = image_np.astype(int).copy()

        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=.3,
            agnostic_mode=False
        )

        plt.figure(figsize=(8, 8))
        plt.imshow(image_np_with_detections)

    plt.show()


# %%

test_detection(
    tfrecords_filename=config.TEST_RECORD,
    tfrecords_num=14,
    detect_fn=exported_detect_fn
)
