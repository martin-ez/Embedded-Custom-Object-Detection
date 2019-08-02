from os import path, makedirs
from shutil import copyfile
from sys import exit
from src.Helpers import load_json, replace
from object_detection import model_hparams
from object_detection import model_lib
import tensorflow as tf

dirname = path.dirname(__file__)
out_model_name = None
config_file = None
config = None
classes = None

def load_configuration():
    global config
    global classes
    print(' ├─ Loading config and classes file')
    config = load_json(path.join(dirname, 'config.json'))
    classes = load_json(path.join(dirname, config['setup_training']['training_class_map'] + '.json'))['classes']

def prepare_model_folder():
    global out_model_name
    print(' ├─ Preparing model folder')
    out_model_path = path.join(dirname, 'models', config['setup_training']['out_model_name'])
    if not path.exists(out_model_path):
        makedirs(out_model_path)
    write_label_map(out_model_path)
    write_config_file(out_model_path)

def write_label_map(out_model_path):
    print(' │   ├─ Writing labelmap file')
    labelmap_path = path.join(out_model_path, 'label_map.pbtxt')
    with open(labelmap_path, 'w') as labelmap:
        for i in range(1, len(classes)):
            labelmap.write('item {\n')
            labelmap.write('    id: '+str(i)+'\n')
            labelmap.write('    name: \''+classes[i]+'\'\n')
            labelmap.write('}\n')
            labelmap.write('\n')

def write_config_file(out_model_path):
    global config_file
    print(' │   ├─ Writing configuration file')
    base_model_path = path.join(dirname, 'models', config['setup_training']['base_model_name'])
    src_config_file = path.join(base_model_path, 'pipeline.config')
    config_file = path.join(out_model_path, 'pipeline.config')
    copyfile(src_config_file, config_file)
    with open(config_file, 'r') as f:
        text = f.readlines()
    edited_text = ''
    replacements = [
        (r'(num_classes:) [0-9]+', r'\1 '+str(len(classes) - 1)),
        (r'PATH_TO_BE_CONFIGURED/model\.ckpt', path.abspath(path.join(base_model_path, 'model.ckpt'))),
        (r'PATH_TO_BE_CONFIGURED/.+\.pbtxt', path.abspath(path.join(out_model_path, 'label_map.pbtxt'))),
        (r'PATH_TO_BE_CONFIGURED/.+train\.record', path.abspath(path.join(dirname, r'data', 'train.record'))),
        (r'PATH_TO_BE_CONFIGURED/.+val\.record', path.abspath(path.join(dirname, r'data', 'test.record')))
    ]
    for line in text:
        edited_text += replace(line, replacements)
    with open(config_file, 'w') as f:
        f.write(edited_text)
    print(' │')

def start_training():
    config = tf.estimator.RunConfig(model_dir=out_model_name)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(None),
        pipeline_config_path=config_file,
        train_steps=100,
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

def main():
    print('---------------------------------')
    print('-       MODEL TRAINING -')
    print('---------------------------------')
    print('- SETTING TRAINING CONFIGURATION')
    load_configuration()
    prepare_model_folder()
    print('- STARTING TRAINING')
    start_training()

if __name__ == '__main__':
    main()
