import  os
import tensorflow as tf
from shutil import copyfile
from google.protobuf import text_format
from tbl_detection.models.research.object_detection.protos import pipeline_pb2
from tbl_detection import config


# %%

def write_pipeline_config(pipeline_config_path, pipeline):
    config_text = text_format.MessageToString(pipeline)
    with tf.io.gfile.GFile(pipeline_config_path, "wb") as f:
        f.write(config_text)


# Adjust pipeline config modification here if needed.
def modify_config(pipeline):
    # Model config.
    pipeline.model.ssd.num_classes = 1
    # Train config.
    pipeline.train_config.batch_size = 8
    pipeline.train_config.fine_tune_checkpoint = (os
                                                  .path
                                                  .sep
                                                  .join([config.MODEL_FOLDER,
                                                        'datasets',
                                                         config.MODEL_NAME,
                                                         'checkpoint',
                                                         'ckpt-0']))

    pipeline.train_config.fine_tune_checkpoint_type = 'detection'

    # Train input reader config.
    pipeline.train_input_reader.label_map_path = config.LABEL_MAP
    pipeline.train_input_reader.tf_record_input_reader.input_path[0] = (config
                                                                        .TRAIN_RECORD)
    # Eval input reader config.
    pipeline.eval_input_reader[0].label_map_path = config.LABEL_MAP
    pipeline.eval_input_reader[0].tf_record_input_reader.input_path[0] = (config
                                                                          .TEST_RECORD)
    return pipeline


def read_pipeline_config(pipeline_config_path):
    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline)
    return pipeline


def clone_pipeline_config():
    copyfile(os.path.sep.join([config.MODEL_FOLDER,
                               'datasets',
                               config.MODEL_NAME,
                               'pipeline.config']),
             r'tbl_detection\pipeline.config')


def setup_pipeline(pipeline_config_path):
    clone_pipeline_config()
    pipeline = read_pipeline_config(pipeline_config_path)
    pipeline = modify_config(pipeline)
    write_pipeline_config(pipeline_config_path, pipeline)
    return pipeline


# %%

# Adjusting the pipeline configuration.
pipeline_config = setup_pipeline(r'D:\holomb_learn\obj_detection\tbl_detection\pipeline.config')
