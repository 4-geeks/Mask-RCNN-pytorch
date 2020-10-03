# Mask-RCNN-pytorch
pytorch version of mask-rcnn based on torchvision model with VOC dataset format

## Training
To start training in your own dataset, in train.py script, add your VOC dataset path to DATASET_PATH variable.
you can label your data with [labelme](https://github.com/wkentaro/labelme) and Exporting VOC-format dataset from json files with [labelme2voc](https://github.com/wkentaro/labelme/tree/master/examples/instance_segmentation).

prepare your dataset in this format:
```
my_dataset
      ├── JPEGImages
      │       ├── file11.ext
      │       └── file12.ext
      │
      ├── SegmentationObject
      │       ├── file21.ext
      │       ├── file22.ext
      │       └── file23.ext
      │
      └── SegmentationClass
              ├── file21.ext
              ├── file22.ext
              └── file23.ext
```
