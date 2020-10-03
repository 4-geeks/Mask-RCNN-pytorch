# Mask-RCNN-pytorch
Pytorch version of mask-rcnn based on torchvision model with VOC dataset format

## Training
To start training in your own dataset, in ```train.py```, add your VOC dataset path to DATASET_PATH variable.

You can label your data with [labelme](https://github.com/wkentaro/labelme) and Export VOC-format dataset from json files with [labelme2voc](https://github.com/wkentaro/labelme/tree/master/examples/instance_segmentation).

Prepare your dataset in this format:
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

## Testing
Add your IMAGE_PATH and MODEL_PATH to ```test.py```  and enjoy!
