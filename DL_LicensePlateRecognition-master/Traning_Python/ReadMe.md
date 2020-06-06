## Prerequisites
``pip install -r requirements.txt``


## Install Darknet
```
git clone https://github.com/pjreddie/darknet.git
cd darknet
mv ../custom darknet
wget https://pjreddie.com/media/files/darknet53.conv.74
```
`darknet53.conv.74` contains the weights got by training the [Imagenet](http://www.image-net.org/). We are going to take these 
weights as initial weights and train our yoloV3 network with config files that contain layers configuration.  

## Run Darknet

If not using CUDA and OpenCV
```
make
```

Otherwise edit MakeFile and enable OpenCV and CUDA if needed and then run
```
OPENCV=1
GPU=1
CUDNN=1

export PKG_CONFIG_PATH=/path/to/the/file
cd darknet
make
```

## Download dataset
Download dataset from these links and place them in the following locations

* Download cars dataset and their annotations from [here](https://drive.google.com/open?id=1UolAVqsxVPzxL2fXYtpG3PI6B4i3dk8v).
  Unzip and place it in `Yolo_LicensePlateDetection/dataset`

* Download License Plates dataset and their annotations from [here](https://drive.google.com/open?id=1ezZ4x7Nv_rk-P1K9LKxl6Z0oKTLNn4Dx).
  Unzip and place it in `Yolo_CharacterSegmentation/dataset`

* Download Characters dataset and their annotations from [here](https://drive.google.com/open?id=1GrbtIwmh8ko1ZVM3lo0MJ62x8VmeVHVE).
  Unzip and place it in `CNN_CharacterRecognition/dataset`


## Training
Then start training the network using the following commands.


### Detection of License Plate, given a car image
Make sure you are in `/darknet` folder when running these commands
```
./darknet detector train ../Yolo_LicensePlateDetection/config/license_plate.data custom/yolov3-license-plates.cfg darknet53.conv.74
```

### Detection of Characters in License plate, given a license plate image
```
./darknet detector train ../Yolo_CharacterSegmentation/config/characters.data ./custom/yolov3_character.cfg darknet53.conv.74
```

### CNN Character detection
```
cd ../CNN_CharacterRecognition
python cnn_train.py
``` 


