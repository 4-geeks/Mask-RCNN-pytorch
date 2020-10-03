# Mask-RCNN-pytorch
pytorch version of mask-rcnn based on torchvision model with VOC dataset format

## Training
To start training in your own dataset, in train.py script, add your VOC dataset path to DATASET_PATH variable.

you can label your data with [labelme](https://github.com/wkentaro/labelme) and Export VOC-format dataset from json files with [labelme2voc](https://github.com/wkentaro/labelme/tree/master/examples/instance_segmentation).

prepare your dataset in this format:
```
my_dataset
      ├── JPEGImages
      │       ├── image1.jpg
      │       └── image2.jpg
      │
      ├── SegmentationObject
      │       ├── image1.png
      │       └── image2.png
      │
      └── SegmentationClass
              ├── image1.png
              └── image2.png
```
