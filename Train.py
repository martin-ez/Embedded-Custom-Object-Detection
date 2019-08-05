from os import path, makedirs
from shutil import copyfile, rmtree
from sys import exit
from src.Helpers import load_json, replace
from object_detection import model_hparams
from object_detection import model_lib
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

dirname = path.dirname(__file__)
out_model_path = None
out_model_ckpt_path = None
config_file = None
config = None
classes = None
training_steps = 0

def load_configuration():
    global config
    global classes
    print(' ├─ Loading config and classes file')
    config = load_json(path.join(dirname, 'config.json'))
    classes = load_json(path.join(dirname, config['training']['training_class_map'] + '.json'))['classes']

def prepare_model_folder():
    global out_model_path
    global out_model_ckpt_path
    print(' ├─ Preparing model folder')
    out_model_path = path.join(dirname, 'models', config['training']['output_model'])
    out_model_ckpt_path = path.join(out_model_path, 'checkpoints')
    if not path.exists(out_model_path):
        makedirs(out_model_path)
    if path.exists(out_model_ckpt_path):
        rmtree(out_model_ckpt_path)
    makedirs(out_model_ckpt_path)
    write_label_map()
    write_config_file()

def write_label_map():
    print(' │   ├─ Writing labelmap file')
    labelmap_path = path.join(out_model_ckpt_path, 'label_map.pbtxt')
    with open(labelmap_path, 'w') as labelmap:
        for i in range(1, len(classes)):
            labelmap.write('item {\n')
            labelmap.write('    id: '+str(i)+'\n')
            labelmap.write('    name: \''+classes[i]+'\'\n')
            labelmap.write('}\n')
            labelmap.write('\n')

def write_config_file():
    global config_file
    global training_steps
    print(' │   ├─ Writing configuration file')
    training_steps = str(config['training']['training_steps'])
    base_model_path = path.join(dirname, 'models', config['training']['base_model'])
    src_config_file = path.join(base_model_path, 'pipeline.config')
    config_file = path.join(out_model_ckpt_path, 'pipeline.config')
    copyfile(src_config_file, config_file)
    with open(config_file, 'r') as f:
        text = f.readlines()
    edited_text = ''
    ckpt_abs_path = replace(path.abspath(path.join(base_model_path, 'model.ckpt')), [(r'\\',r'/')])
    labelmap_abs_path = replace(path.abspath(path.join(out_model_ckpt_path, 'label_map.pbtxt')), [(r'\\',r'/')])
    trainrecord_abs_path = replace(path.abspath(path.join(dirname, r'data', 'train.record')), [(r'\\',r'/')])
    testrecord_abs_path = replace(path.abspath(path.join(dirname, r'data', 'test.record')), [(r'\\',r'/')])
    replacements = [
        (r'(num_classes:) [0-9]+', r'\1 '+str(len(classes) - 1)),
        (r'(num_steps:) [0-9]+', r'\1 '+training_steps),
        (r'PATH_TO_BE_CONFIGURED/model\.ckpt', ckpt_abs_path),
        (r'PATH_TO_BE_CONFIGURED/.+\.pbtxt', labelmap_abs_path),
        (r'PATH_TO_BE_CONFIGURED/.+train\.record', trainrecord_abs_path),
        (r'PATH_TO_BE_CONFIGURED/.+val\.record', testrecord_abs_path)
    ]
    for line in text:
        edited_text += replace(line, replacements)
        if 'feature_extractor' in line:
            edited_text += '      override_base_feature_extractor_hyperparams: true\n'
    with open(config_file, 'w') as f:
        f.write(edited_text)
    print(' │')

def start_training():
    config = tf.estimator.RunConfig(model_dir=out_model_ckpt_path)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(None),
        pipeline_config_path=config_file,
        train_steps=None,
        sample_1_of_n_eval_examples=1,
        sample_1_of_n_eval_on_train_examples=(5))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

def freeze_graph():
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(config_file, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge('', pipeline_config)
    exporter.export_inference_graph(
        'image_tensor', pipeline_config, path.join(out_model_ckpt_path, 'model.ckpt-'+training_steps),
        out_model_path, input_shape=None,
        write_inference_graph=False)

def main(_):
    print('---------------------------------')
    print('-       MODEL TRAINING -')
    print('---------------------------------')
    print('- SETTING TRAINING CONFIGURATION')
    load_configuration()
    prepare_model_folder()
    print('- STARTING TRAINING')
    start_training()
    print('- FREEZING GRAPH')
    freeze_graph()

if __name__ == '__main__':
    tf.app.run()
