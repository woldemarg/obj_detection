import os

# %%

# download and unzip original dataset from here
# https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/food-datasets/Food-5K.zip
ORIG_INPUT_DATASET = r'transfer_learning_via_DL\cnn_as_feature_extractor\Food-5K'
BASE_PATH = r'transfer_learning_via_DL\cnn_as_feature_extractor\dataset'

# %%

TRAIN = 'training'
TEST = 'evaluation'
VAL = 'validation'

# %%

CLASSES = ['non_food', 'food']
BATCH_SIZE = 32

# %%

BASE_CSV_PATH = r'transfer_learning_via_DL\cnn_as_feature_extractor\output'
LE_PATH = os.path.sep.join([BASE_CSV_PATH, 'le.cpickle'])
MODEL_PATH = os.path.sep.join([BASE_CSV_PATH, 'model.cpickle'])

# %%

INPUT_SIZE = (224, 224)
