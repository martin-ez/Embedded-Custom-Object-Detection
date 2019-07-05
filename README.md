# Embedded Custom Object Detection

___
## Setup
As mention previously, this project aims to use a Raspberry Pi to detect objects, therefore this documentation is aim to setup the environment on that device. However, is not practical to do the model training on the Raspberry, so we encourage you to setup the environment on a machine that could use GPU to accelerate the training.

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

OpenCV, Protobuf and Tensorflow need to be installed in the device. User EdjeElectronics on Github writed [this detailed tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi) on how to install them. Follow steps 1 through 5.

### Base Pre-trained Model Setup
Inside the tensorflow directory cloned previously, navigate to `tensorflow/models/research/object_detection`.
Clone this repository using the following command:
```
git clone https://github.com/martin-ez/Embedded-Custom-Object-Detection.git ECOD
```
Install the project dependencies:
```
pip3 install -r ECOD/requirements.txt
```
Finally, run the Setup script to download the base pre-trained model:
```
python3 ECOD/Setup.py
```
You should see a message saying the model is ready to use and the path where is located.

### Select another base model (Optional)
The previous step will download the pre-trained ssd_mobilenet_v2 model, which is lite and fast for devices with low resources like the Raspberry. If you desire to use a different model, the project is compatible with any of the models included in [TensorFlow's Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
To download one of these models, modified the configuration file `ECOD/config.json`, changing the 'base_model_name' for a name to identify the new model and copying the download URL or the selected model (right-click > Copy link address)
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
python3 ECOD/Setup.py
```
A new folder in the `ECOD/models` location should be created with the selected model inside.

## Test pre-trained model
Once the environment have been setup, you can test that the image detection is working correctly by using:
```
python3 ECOD/Detect.py ---image <Path-to-test-image> --out_folder <Path-to-output-image>
```
Where `<Path-to-test-image>` is the location of an image to be detected and `<Path-to-output-image>` is a folder where the resulting image will be save. (Alternatively, you can omit the `--image` property to start a continuous detection taking images from a webcam connected to the device.)
You should see a message of the objects found in the image, and a image with the bounding-boxes of them should have been saved at the specified location, with a timestamp as the name.

## Training with custom dataset
TODO

## Detecting objects with a webcam
You can test the trained model on any image as we mentioned previously:
```
python3 ECOD/Detect.py ---image <Path-to-test-image> --out_folder <Path-to-output-image>
```

However, this project is design to do a continuous detection using a webcam, sending the results to a remote server. To adjust this behaviour you can modify the configuration file `ECOD/config.json`:
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
- `model_name` the model to be used for detection. This should be a name of a folder in the 'ECOD/models' folder.
- `sample_rate` the time between each detection. This can be declare on seconds (s) or milliseconds (ms).
- `conf_threshold` the minimum confidence the network should output on an object to be considered a detection. A value in range 0 - 1.
- `api_url` the URL of the rest service that will receive the results.

Once the configuration is set, the detection process can start by running:
```
python3 ECOD/Detect.py
```

### Test server
For testing purposes, a test server is included to intercept the detection results. It can be started by running:
```
python3 ECOD/src/TestServer.py
```
This will listen to POST requests on `http://127.0.0.1:5000/detection`, displaying the results and saving the images on `ECOD/src/server_out/`.

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
