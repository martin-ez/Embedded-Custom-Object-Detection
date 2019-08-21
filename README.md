# Embedded Custom Object Detection

A wrapper around Tensorflow's Object Detection API, which facilitates setting up object detection models, re-training them to any set of classes and running them on IoT and embedded devices like the Raspberry Pi. The project provides four utility scripts, which are easily configurable from a `config.json` configuration file.

- **Setup.py:** Downloads and prepares models from Tensorflow's Detection Model Zoo to be used in inference.
- **Dataset.py:** Converts a custom dataset, pre-labeled to PascalVOC format in XML files, to Tensorflow TFRecord format.
- **Train.py:** Prepares and runs a training session to perform transfer learning from one pre-trained model to a new one using a custom dataset.
- **Detect.py:** Runs a model to perform inference to an image or using a webcam videofeed.

The following documentation will explain how to use each script and the options it provides.

This was developed as part of a University of Los Andes project, that intends to create an interactive dashboard to monitor an aquaponics system. Using the tools provided, the dashboard will detect fish and plants in the system to record their movement and growth over time.

## Table of contents

- [Environment setup](#environment-setup)
  * [Prerequisites](#prerequisites)
  * [Project structure](#project-structure)
  * [Tensorflow's Object Detection API installation](#tensorflow-s-object-detection-api-installation)
  * [Installing project dependencies](#installing-project-dependencies)
- [Models folder structure](#models-folder-structure)
- [Setup.py](#setuppy)
  * [Base pre-trained model setup](#base-pre-trained-model-setup)
  * [Select another base model (Optional)](#select-another-base-model--optional-)
  * [Test pre-trained model](#test-pre-trained-model)
- [Dataset.py](#datasetpy)
  * [Training with custom dataset](#training-with-custom-dataset)
  * [Collect dataset images](#collect-dataset-images)
  * [Label dataset images](#label-dataset-images)
  * [Writing class map file](#writing-class-map-file)
  * [Generate TFRecords files](#generate-tfrecords-files)
- [Train.py](#trainpy)
  * [Preparing and running training process](#preparing-and-running-training-process)
  * [Recovering from training session failures](#recovering-from-training-session-failures)
  * [Exporting trained model to Raspberry Pi](#exporting-trained-model-to-raspberry-pi)
- [Detect.py](#detectpy)
  * [Test server](#test-server)
  * [Results format](#results-format)
- [Reference](#reference)

## Environment setup
The end goal of the project is to run the trained detection model in IoT devices like the Raspberry Pi. However, the training of the model is a resource demanding task, so a more capable computer should be used in that step. For that reason, it's best if you set up the project environment in both of the devices. For  simplicity, we will focus on the preparation of the Raspberry Pi, since it's almost the same as in any Linux machine. The main difference is the installation of Tensorflow, where for the training machine you can install tensorflow-gpu if you have a CUDA enabled GPU to accelerate the training process.

### Prerequisites

Due to the unusual architecture of the Raspberry Pi's processor, we will need to compile some of the main dependencies from source code. Unfortunately, this process takes quite a long time, between 1 to 2 hours for all the packages.

First update Raspberry Pi packages and firmware:
```
sudo apt-get update
sudo apt-get upgrade
```

#### Python 3.5
Tensorflow only supports Python3.5 on the Raspberry Pi's architecture, so we need to use this version of Python.

Install all the needed dependencies first:
```
sudo apt-get install build-essential libc6-dev
sudo apt-get install libncurses5-dev libncursesw5-dev libreadline6-dev
sudo apt-get install libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev
sudo apt-get install libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev
sudo apt-get install python3-dev
```
Download and unzip the Python source code:
```
wget https://www.python.org/ftp/python/3.5.7/Python-3.5.7.tgz
tar -zxvf Python-3.5.7.tgz
cd Python-3.5.7
```
Compile and install, this might take quite a long time due to the Raspberry performance. Run:
```
./configure
make -j4
sudo make install
```
Finally, check that the installation was successful and, optionally, delete the folder to save space:
```
python3 --version
cd ..
sudo rm -fr ./Python-3.5.7*
```

#### OpenCV

We gonna use OpenCV to manage our video feed from the webcam. We need to install some image and video codex dependencies to install the library on the device.

You can install them running:
```
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install qt4-dev-tools
```
And then install OpenCV using pip:
```
sudo pip3 install opencv-contrib-python
```

#### Protobuf

The Object Detection API uses Google's Protocol Buffers, or protobuf. This library allows Tensorflow to serialize data quickly.

Download and unzip the tar file:
```
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.9.1/protobuf-all-3.9.1.tar.gz
tar -vxzf ./protobuf-3.9.1.tar.gz
cd ./protobuf-3.9.1/
```
Download dependencies and prepare the compilation:
```
sudo apt-get install autoconf libtool
./autogen.sh
./configure
```
Compile and install protobuf, these steps could take quite a long time to complete:
```
make -j4
sudo make install
sudo ldconfig
```
Check that the installation completed correctly:
```
protoc --version
```

### Project structure
We will create the folder structure for the project. First create a root folder (You can name this folder however you'd like):
```
cd ~
mkdir Pi_Object_Detection
cd Pi_Object_Detection
```
Next clone the Tensorflow's model repository inside the root folder:
```
git clone --recurse-submodules https://github.com/tensorflow/models.git
```
Clone this repository in the root folder as well:
```
git clone https://github.com/martin-ez/Embedded-Custom-Object-Detection.git ECOD
```
You should have the following structure:
```
Pi_Object_Detection
├─ models
│   ├─ official
│   ├─ research
│   ├─ samples
│   └─ tutorials
└─ ECOD
    ├─ src
    ├─ Setup.py
    ├─ Dataset.py
    ├─ Train.py
    ├─ Detect.py
    └─ config.json
```
### Tensorflow's Object Detection API installation
Next step is to compile the protobuf files that tensorflow uses for their object detection API, to do this navigate to the research folder and run the protobuf compiler:
```
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
```
We also need to install the object detection library, running:
```
python3 setup.py build
sudo python3 setup.py install
```
And then add the research and slim folders to the python path: (Replace `<PATH_TO_ROOT_FOLDER>` with the absolute path of the root folder, for example `C:/Documents/Pi_Object_Detection`)
```
export PYTHONPATH=$PYTHONPATH:<PATH_TO_ROOT_FOLDER>/models/research:<PATH_TO_ROOT_FOLDER>/models/research/slim
```
### Installing project dependencies
Navigate back to the root folder and into this project repository:
```
cd ../..
cd ECOD
```
Install the project dependencies:
```
sudo pip3 install -r requirements.txt
```

## Models folder structure

The project stores the different models on a `models` folder, which will be created when running the `Setup.py` script for the first time. Inside this folder, each model will have a folder with the name of the model as its name, inside it you will find the frozen inference graph to use in detection and the model checkpoint to use as a start point in training. You will also see a pipeline configuration file that describes the model training environment and a checkpoints folder if the model is an output of a training session.

The folder structure will look something like this:
```
ECOD
├─ ...
└─ models
    ├─ name_model_1
    │   ├─ checkpoints
    │   ├─ saved_model
    │   ├─ checkpoint
    │   ├─ frozen_inference_graph.pb
    │   ├─ model.ckpt.data-00000-of-00001
    │   ├─ model.ckpt.index
    │   ├─ model.ckpt.meta
    │   └─ pipeline.config
    ├─ name_model_2
    │   └─ ...
    └─ ...
```

## Setup.py

Usage:
```
python3 Setup.py
```

Configuration in `config.json`:
```
{
  "setup": {
    "base_model_name": "ssd_inception_v2",
    "base_model_url": "http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz"
  },
  ...
}
```
- `base_model_name` name of the model that will be downloaded.
- `base_model_url` download link to the tar.gz file containing the pre-trained model.

### Base pre-trained model setup

Run the Setup script to download the base pre-trained model:
```
python3 Setup.py
```
This will download the model and be located in a new folder, which contains everything needed to run the model:
```
ECOD
├─ models
│ ├─ ssd_inception_v2_coco
│ │ ├─ saved_model
│ │ ├─ checkpoint
│ │ ├─ frozen_inference_graph.pb
│ │ ├─ model.ckpt.data-00000-of-00001
│ │ ├─ model.ckpt.index
│ │ ├─ model.ckpt.meta
│ │ └─ pipeline.config
```

### Select another base model (Optional)
The previous step will download the pre-trained ssd_inception_v2 model, which gives a good compromise between speed and accuracy. If you desire to use a different model, the project is compatible with any of the COCO-trained 'boxes' models included in [TensorFlow's Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
To download one of these models, modify the configuration file `config.json`, changing the 'base_model_name' to a name that will identify the new model and copying the download URL or the selected model (right-click > Copy link address)
```
"setup": {
  "base_model_name": "<MODEL_IDENTIFIER>",
  "base_model_url": "<MODEL_DOWNLOAD_URL>"
}
```
Run the Setup script once again:
```
python3 Setup.py
```
A new folder in the `models` location should be created with the selected model inside.

### Test pre-trained model
Once the environment has been set up, you can test that the image detection is working correctly by using:
```
python3 Detect.py ---image <Path-to-test-image>
```
Where `<Path-to-test-image>` is the location of an image to be detected. (Alternatively, you can omit the `--image` property to start a continuous detection taking images from a webcam connected to the device.)
You should see a message on console of the objects found in the image, and an image with the bounding-boxes placed on them.

## Dataset.py

Usage:
```
python3 Datatset.py
```

### Training with a custom dataset
The models featured on TensorFlow's Detection Model Zoo are trained on the COCO (Common Objects in Context) dataset, which has 80 different classes including person, car, cat, dog and others. We will use a technique known as Transfer Learning to re-purpose an already trained model so it can detect new classes. This practice has proven to reduce training time and the need for bigger datasets, since the convolution network already has learned to detect basic shapes and patterns.

A model learns to detect new classes using pre-classified images as its training data. This way, the human knowledge used to classify this images is transfer into the model weights and parameters. This means that you will have to prepare a dataset for the model to train with. There are three steps to train the model with our custom data: collect images, label the images and train the model.

### Collect dataset images
The first step is to collect as many images as possible containing the classes we want to identify. Depending on the complexity and number of classes, the number of images required for a proper training may vary. Normally, 150 to 200 images is an acceptable minimum, but the bigger the training dataset, the better the model can perform. It's also a good idea to use a wide range of images, using different light conditions, different backgrounds, showcasing the classes at different angles and distances, etc. This will help the model to generalize better, and therefore be able to predict the classes on an image more accurately.

After you collect all the images, you will have to divide them between a training set and a test set. The common way to do this is to select 10% of the images at random to be the testing set and the rest is used for training. This is needed so we have a set of test images that the model will not see during training and will work as a way to measure the training process performance.

Using the laptop or desktop computer you will perform the training on, copy the images you collected inside the `data` folder of the project. Putting the train and test images in there respective folders.
```
ECOD
├─ ...
└─ data
  ├─ Test
  └─ Train
```

### Label dataset images
We need to label each image, identifying each class instance present in the image and drawing its bounding box. To do this, we can use tzutalin's [labelImg](https://github.com/tzutalin/labelImg) utility. This tool allow you to open a directory and navigate each image with a predefined set of classes, making the labeling process more easy. It will generate an XML file with the data in PascalVOC format for each image you label, saving them alongside the images automatically.

The utility is available in PyPI, so you should be able to install it by running:
```
sudo pip3 install labelimg
```
But if a problem occurs you can follow the repo instructions.

Then run it issuing:
```
labelImg
```
The GUI of the utility should appear. Open the folder with the images and start labeling them.


### Writing class map file

When all the training and testing images are labeled, we will create a class map file to define the classes that the model will train on. This file must be in a json format and be located in the `data` folder. Included in this repo is a class map for the COCO dataset, named `COCO_classes.json` you can use it as a guide. For example, a class map for a datatset of Piano, Guitar and Drumsticks could look like:
```
{
  "classes": [
    "undefined",
    "piano",
    "guitar",
    "drumstick"
  ]
}
```
Two things to keep in mind when writing the class map is that the first position always needs to be a default or undefined class, and the label names should be exactly as you wrote them on the labelImg utility.

Open the configuration file, `config.json`, and write the name of the class map you created on the `training_class_map` field, omitting the .json extension:
```
{
  ...
  "training": {
    ...
    "training_class_map": "<CUSTOM_CLASS_MAP>"
  },
  ...
}
```

### Generate TFRecords files

Finally, we will generate the Tensorflow records, which compiles all the information about the dataset in two `.record` files. To do this simply run the Dataset script:
```
python3 Datatset.py
```
This will read all the XML and images and write the TFRecords files. It will also give you a brief summary of the dataset, showing the number of instances for each class in the two sets of images. This information is useful since an unbalanced dataset could lead to problems in training.

## Train.py

Usage:
```
python3 Train.py [--freeze <CHECKPOINT>]
```
- `freeze` Number of the checkpoint to be frozen. When this flag is included the training process won't start, only the freeze graph process.

Configuration in `config.json`:
```
{
  ...
  "training": {
    "output_model": "<CUSTOM_MODEL>",
    "base_model": "ssd_inception_v2",
    "training_steps": 22000,
    "training_class_map": "<CUSTOM_CLASS_MAP>"
  },
  ...
}
```
- `output_model` name of the model that will be generated with the training session.
- `base_model` name of the model that will be used as a starting point for the training.
- `training_steps` number of training steps the model will perform.
- `training_class_map` name of class map of the dataset used for training, located on the 'data' folder and omitting the .json extension.

### Preparing and running training process

As mention previously, the training process demands a lot of computing power, so it's not a good idea to perform the training on the Raspberry Pi. Follow the installation steps on a laptop or desktop computer so the training can be done in that machine. The training process can be significantly sped up using a GPU, if the chosen computer has a CUDA enabled GPU you can uninstall tensorflow and install tensorflow-gpu, CUDA and cuDNN. You can know more about this setup on [Tensorflow's official guide](https://www.tensorflow.org/install/gpu).

Modify the `config.json` configuration file, selecting the base model, the output model, the number of training steps and the class map file.

Start the training session with:
```
python3 Train.py
```

Depending on the number of steps, this could take several hours to complete. Every 100 steps, you should see a message on console showing the step number and the current loss of the model. In general, we want a loss below 1.0 and it could take 20,000 steps or more to achieve it. This may vary depending on the base model chosen and the size and quality of the dataset.

### Recovering from training session failures

Due to the demanding nature of the training process, computers may crash in the middle of a training session, not letting the script finish the process. However, you can recover from this without losing the training work done by the model. There's two steps to achieve this:

#### Freezing model from checkpoint

Go into the output model folder and look inside its checkpoints folder. Find the last checkpoint saved, to do this look for the largest XXX number that has an index and data file. For example:
```
ECOD
├─ models
│ ├─ ssd_inception_v2_coco
│ │ ├─ checkpoints
│ │ │   ├─ model.ckpt.data-00000-of-00001
│ │ │   ├─ model.ckpt-100.index
│ │ │   ├─ model.ckpt-100.meta
│ │ │   ├─ model.ckpt-220.index
│ │ │   ├─ model.ckpt-220.meta
│ │ │   ├─ model.ckpt-428.index
│ │ │   └─ model.ckpt-428.meta
│ │ └─ ...
```
In this case, the last saved checkpoint was 428. Once you identify this, run the freeze graph process with:
```
python3 Train.py --freeze <CHECKPOINT>
```
Where `<CHECKPOINT>` is the number of the last saved checkpoint you found. This process will create the frozen_inference_graph.pb file and everything else needed to run the model for inference.

#### Resume a training session

After a training session is completed or you freeze the last saved checkpoint of a model, you can create a new session to keep training the model where it left off. To do this, simply modify the `config.json` configuration file to use the trained model as the new base model for the session and write a new identifier for the output model. For example:

```
{
  ...
  "training": {
    "output_model": "custom_mode_v2",
    "base_model": "custom_model",
    ...
  },
  ...
}
```
Set the training steps and verify that it's using the correct class map, then start the training again with:
```
python3 Train.py
```

### Exporting trained model to Raspberry Pi

After training, exporting the model to the Raspberry PI is as easy as copying the resulting model folder and the class map. This contains everything needed to run the model in inference.
Copy the model folder inside the `models` folder and the class map inside the `data` folder. Modify the `config.json` configuration file to point to the model and class map and you are good to go.

## Detect.py

Usage:
```
python3 Detect.py [--image <PATH_TO_IMAGE>] [--output <PATH_TO_OUTPUT_FOLDER>] [--sendresults] [--dontshow]
```
- `image` path to the image that will be detected. When omitted, a continuous detection using a web camera will begin.
- `output` path to a folder where the resulting images will be saved.
- `sendresults` when included, the detection process will send the results of the detection to the Rest API found on the config.json file.
- `dontshow` when included, the detection images will not be shown on the screen.

Configuration in `config.json`:
```
{
  ...
  "inference" : {
    "model_name": "ssd_inception_v2",
    "class_map": "COCO_classes",
    "sample_rate": "5s",
    "conf_threshold": 0.5,
    "api_url": "http://127.0.0.1:5000/detection"
  }
}
```
- `model_name` identifier of the model to be used for detection. This should be a name of a folder in the 'models' folder.
- `class_map` name of class map used by the model, located on the 'data' folder and omitting the .json extension.
- `sample_rate` the time between each detection. This can be declared in seconds (s) or milliseconds (ms).
- `conf_threshold` minimum confidence value the model should output on an object to be considered a detection. A value in range 0 - 1.
- `api_url` URL of the rest service where the model will send the results.

Right now, the detection process only supports USB web cameras, not PiCamera.

### Test server
For testing purposes, a test server is included to intercept the detection results. It can be started by running:
```
python3 src/TestServer.py
```
This will listen to POST requests on `http://127.0.0.1:5000/detection`, displaying the results and saving the images on `src/server_out/`.

### Results format
Results of the detection process will be sent to the server with three objects:
- `raw_image` file with the image as was taken by the webcam.
- `detection_image` file with the image with the bounding-boxes of the detections.
- `json` with the detection results following the next format:
```
{
  "timestamp": "<detection-timestamp>",
  "result": [
    {
      "class": "<Class_1>",
      "instances": [
        {
          "box": [left, right, top, bottom],
          "score": 0.92
        },
        {
          "box": [left, right, top, bottom],
          "score": 0.87
        },
        ...
      ],
    },
    {
      "class": "<Class_2>",
      "instances": [
        {
          "box": [left, right, top, bottom],
          "score": 0.94
        },
        ...
      ],
    },
    ...
  ]
}
```

## Reference
- [Tensorflow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [EdjeElectronics's Tutorial to set up TensorFlow Object Detection API on the Raspberry Pi](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi)
- [tzutalin's Label Image Util](https://github.com/tzutalin/labelImg)
