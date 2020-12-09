import tensorflow as tf
import tensorflow.python.keras.layers.preprocessing

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "Food-11"
# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "dataset"

mnist = tf.keras.datasets.mnist

tf.__version__
