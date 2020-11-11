# Mask-RCNN-pytorch
Pytorch version of mask-rcnn based on torchvision model with VOC dataset format

## Training
To start training in your own dataset, in ```train.py```, add your VOC dataset path to ```DATASET_PATH``` variable.

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

Use this line of code to train
```
$ python3 train.py --data my_dataset --num_classes 11 --num_epochs 150
```

## Testing

Use this line of code to test on your image
```
$ python3 test.py --img test_img.jpg --model ./maskrcnn_saved_models/mask_rcnn_model.pt
```
