import os

# %%

# https://github.com/DevashishPrasad/CascadeTabNet
ORIG_BASE_PATH = r'D:\holomb_learn\obj_detection\tbl_detection\cells'
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, 'cells_images'])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, 'cells_annots'])
TBLS_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, 'tbls_annots'])
TBLS_IMAGES = os.path.sep.join([ORIG_BASE_PATH, 'tbls_images'])
XML_STYLE = r'D:\holomb_learn\obj_detection\tbl_detection\cells\xml_style.xml'
TRAIN_SET = os.path.sep.join([ORIG_BASE_PATH, 'train_set'])
TEST_SET = os.path.sep.join([ORIG_BASE_PATH, 'test_set'])
VAL_SET = os.path.sep.join([ORIG_BASE_PATH, 'val_set'])
LABEL_MAP = r'D:\holomb_learn\obj_detection\tbl_detection\cells\label_map.pbtxt'
LABELS_CSV = os.path.sep.join([ORIG_BASE_PATH, 'labels_csv'])
TF_RECORDS = os.path.sep.join([ORIG_BASE_PATH, 'tf_records'])
TRAIN_RECORD = os.path.sep.join([TF_RECORDS, 'train.record'])
TEST_RECORD = os.path.sep.join([TF_RECORDS, 'test.record'])
TRAIN_CSV = os.path.sep.join([LABELS_CSV, 'train.csv'])
TEST_CSV = os.path.sep.join([LABELS_CSV, 'test.csv'])
LOGS = r'D:\holomb_learn\obj_detection\tbl_detection\train_logs'
MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
MODEL_FOLDER = r'D:\holomb_learn\obj_detection\tbl_detection\model_data'
