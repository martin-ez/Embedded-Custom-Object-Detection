from os import path, makedirs, rename
from urllib import request
from io import BytesIO
import tarfile as tgz
from sys import exit
from src.Helpers import load_json
import progressbar

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def main():
    print('---------------------------------')
    print('        - MODEL SETUP -')
    print('---------------------------------')
    print(' - LOADING CONFIGURATION FILE')
    dirname = path.dirname(__file__)
    config = load_json(path.join(dirname, 'config.json'))['setup']
    model_name = config['base_model_name']
    download_url = config['base_model_url']
    base_name = path.basename(download_url).split('.')[0]
    print(' ├─ Configuration loaded correctly')
    out_path = path.join(dirname, 'models')
    if not path.exists(out_path):
        makedirs(out_path)
        print(' ├─ Models folder created')
    print(' │ ')
    model_path = path.join(out_path, model_name)
    if path.exists(model_path):
        exit(' ├─ !! ERROR: A model already exist in path ' +  model_path)
    print(' - DOWNLOADING MODEL')
    model_data = request.urlretrieve(download_url, filename=None, reporthook=show_progress)[0]
    zipped_model = tgz.open(model_data)
    zipped_model.extractall(out_path)
    zipped_model.close()
    rename(path.join(out_path, base_name), model_path)
    print(' ├─ Model downloaded correctly')
    print(' │ ├─', model_name, 'ready to use')
    print(' │ ├─ Model Path:', model_path)
    print(' │ ')

if __name__ == '__main__':
    main()
