import os
import cv2
import argparse
import matplotlib.pyplot as plt
from mask_rcnn import segmentation_model, plot_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='test_image.jpg', help='path to your test image')
    parser.add_argument('--labels', type=str, default='./my_dataset/labels.txt', help='path to labels list text file (labels.txt)')
    parser.add_argument('--model', type=str, default='./maskrcnn_saved_models/mask_rcnn_model.pt', help='path to saved model')

    args = parser.parse_args()
    
    IMAGE_PATH = args.img
    MODEL_PATH = args.model
    LABEL_PATH = args.labels
    
    with open(LABEL_PATH,'r') as f:
        lines = [line.rstrip() for line in f]
    assert lines[0] == '__ignore__', """first line of labels file must be  \
                                        "__ignore__" (labelme labels.txt)"""
    lines.pop(0) # remove first elements [__ignore__]
    
    num_classes = len(lines)
    classes = dict(zip(range(num_classes),lines))
    
    image = cv2.imread(IMAGE_PATH)
    model = segmentation_model(MODEL_PATH,num_classes)
    pred = model.detect_masks(image, rgb_image=False)   # rgb_image=False if loading image with cv2.imread()

    plotted = plot_masks(image,pred,classes)
    
    os.makedirs('./results', exist_ok=True)
    cv2.imwrite('./results/res.jpg', plotted)

    plt.figure(figsize=(16,12))
    plt.imshow(plotted)
    plt.show()