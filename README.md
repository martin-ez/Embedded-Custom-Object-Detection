# Embedded Custom Object Detection

## Setup
This project aims to use a Raspberry Pi to detect objects, therefore this documentation is aim to setup the environment on that device. However, is not practical to do the model training on the Raspberry, so we encourage you to setup the environment on a machine that could use GPU to accelerate the training.

### Prerequisites
Make sure to have installed a version of Python 3 and pip on the raspberry. Python 3.6 can be installed following the next commands:
```
sudo apt-get install python3-dev libffi-dev libssl-dev -y
wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tar.xz
tar xJf Python-3.6.3.tar.xz
cd Python-3.6.3
./configure
make
sudo make install
sudo pip3 install --upgrade pip
```

OpenCV, Protobuf and Tensorflow need to be installed in the device. User EdjeElectronics on Github writed [this detailed tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi) on how to install them, you can follow steps 1 through 4. At the end of step 4 you will be instruct to reboot the Raspberry Pi, do so and then return to this file.

### Project structure
We will create the folder structure for the project. First create a root folder:
```
mkdir Pi_Object_Detection
cd Pi_Object_Detection
```
You can name this folder however you like. Next clone the Tensorflow's model repository inside the root folder:
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
pip3 install --user -r requirements.txt
```

### Base pre-trained model setup
Run the Setup script to download the base pre-trained model:
```
python3 Setup.py
```
This will download the model and located in a new folder, which contains everything needed to run the model:
```
Pi_Object_Detection
├─ models
│   ...
└─ ECOD
    ├─ models
    │ ├─ ssd_inception_v2_coco
```

### Select another base model (Optional)
The previous step will download the pre-trained ssd_inception_v2 model, which is lite and fast for devices with low resources like the Raspberry. If you desire to use a different model, the project is compatible with any of the models included in [TensorFlow's Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
To download one of these models, modified the configuration file `config.json`, changing the 'base_model_name' for a name to identify the new model and copying the download URL or the selected model (right-click > Copy link address)
```
...
  "setup_training": {
    "out_model_name": "custom_model",
    "base_model_name": "<Model identifier>",
    "base_model_url": "<Selected model download URL>"
  }
}
```
Run the Setup script once again:
```
python3 Setup.py
```
A new folder in the `models` location should be created with the selected model inside.

## Test pre-trained model
Once the environment have been setup, you can test that the image detection is working correctly by using:
```
python3 Detect.py ---image <Path-to-test-image> --out_folder <Path-to-output-image>
```
Where `<Path-to-test-image>` is the location of an image to be detected and `<Path-to-output-image>` is a folder where the resulting image will be save. (Alternatively, you can omit the `--image` property to start a continuous detection taking images from a webcam connected to the device.)
You should see a message of the objects found in the image, and a image with the bounding-boxes of them should have been saved at the specified location, with a timestamp as the name.

## Training with custom dataset
The models feature on TensorFlow's Detection Model Zoo are trained on the COCO (Common Objects in Context) dataset, which 80 different classes including person, car, cat, dog and others. We will use a technique known as Transfer Learning to re-purpose an already trained model so it can detect new classes. This technique have proven to reduce training time and the need for bigger datasets, since the convolution network already learned to detect basic shapes and patterns.

A model learns to detect new classes using pre-classified images as its training data. This way, the human knowledge use to classify this images is transfer into the model weights and parameters. This means that you will have to prepare a dataset for the model to train on. There are three steps to train the model with our custom data: collect images, label the images and train the model.

As mention previously, the training process demands a lot of computing power, so its not a good idea to perform the training on the Raspberry Pi. Follow the installation steps in a laptop or desktop computer so the training can be done in that machine. The training process can be significantly speed up using a GPU, if the chosen computer has one you can uninstall tensorflow and install tensorflow-gpu, CUDA y cuDNN. You can know more about this setup on [Tensorflow's official guide](https://www.tensorflow.org/install/gpu).

### Collect dataset images
The first step is to collect as many images as possible containing the classes we want to identify. Depending on the complexity and number of classes, the number of images required for a good training may vary. Normally, 150-200 images is a good minimum. But the bigger the training dataset, the better the model can perform.
Is also a good idea to use a wide range of images, using different light conditions, different backgrounds, showcasing the classes at different angles and distances, etc. This will help the model to generalize better, and therefore be able to predict the classes on an image more accurately.

After you collect all the images, you will have to divide them between a training set and a test set. The common way to do this is to select 10% of the images at random to be the testing set and the rest is use for training. This is needed so we have a set of test images that the model will not see during training and will work as a way to measure the training process performance.

Create a new folder inside the ECOD folder named data, and inside this folder create a Test and Train folders. You will end up with a structure like this:
```
Pi_Object_Detection
├─ models
│   ...
└─ ECOD
    ├─ ...
    └─ data
      ├─ Test
      └─ Train
```
Copy the image sets in their respective folders.

### Label dataset images
We require to label each image, identifying each class instance present in  the image and drawing its bounding box. To do this, we can use tzutalin's [labelImg](https://github.com/tzutalin/labelImg) utility. This tools allow you to open a directory and navigate each image with a predefined set of classes, making the labeling process more easy. It will generate an XML file for each image you label, saving them alongside the images automatically.

The utility should have been installed in the first step and can be run issuing the command:
```
labelImg
```
But if a problem occurs you can follow the repo instructions.

When all the training and testing image are label, we will create a class map file to define the classes that the model will train on. This file must be in a json format and be located in the root of the `ECOD` folder. Included on the repo is a class map for the COCO dataset, named `COCO_classes.json` you can use it as a guide. For example, a class map for a datatset of Piano, Guitar and Drumsticks could look like:
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
Open the configuration file, `config.json`, and write the name of the `training_class_map`:
```
{
  "inference" : {
    ...
  },
  "setup_training": {
    ...
    "training_class_map": "COCO_classes",
  }
}
```

Finally, we will generate the Tensorflow records, which compiles all the information about the dataset in two `.record` files. To do this simply run the Dataset script:
```
python3 Datatset.py
```
This will read all the images and write the tf records files. It will also give you a brief summary of the dataset, showing the number of instances for each class in the two sets of images.

### Train the model
TO-DO

## Detecting objects with a webcam
You can test the trained model on any image as we mentioned previously:
```
python3 Detect.py ---image <Path-to-test-image> --out_folder <Path-to-output-image>
```

However, this project is design to do a continuous detection using a webcam, sending the results to a remote server. To adjust this behaviour you can modify the configuration file `config.json`:
```
{
  "inference" : {
    "model_name": "mobilenet_v2",
    "sample_rate": "1s",
    "conf_threshold": 0.5,
    "api_url": "http://127.0.0.1:5000/detection"
  },
...
```
- `model_name` the model to be used for detection. This should be a name of a folder in the 'models' folder.
- `sample_rate` the time between each detection. This can be declare on seconds (s) or milliseconds (ms).
- `conf_threshold` the minimum confidence the network should output on an object to be considered a detection. A value in range 0 - 1.
- `api_url` the URL of the rest service that will receive the results.

Once the configuration is set, the detection process can start by running:
```
python3 Detect.py
```

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
  "result": {
    "<Class_1>": [
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
    "<Class_2>": [
      {
        "box": [left, right, top, bottom],
        "score": 0.94
      },
      ...
    ],
    ...
  }
}
```

## Reference
- [Tensorflow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [EdjeElectronics's Tutorial to set up TensorFlow Object Detection API on the Raspberry Pi](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi)
- [tzutalin's Label Image Util](https://github.com/tzutalin/labelImg)
