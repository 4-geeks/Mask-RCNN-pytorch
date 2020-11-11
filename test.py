import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mask_rcnn import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='test_image.jpg', help='path to your test image')
    parser.add_argument('--model', type=str, default='./maskrcnn_saved_models/mask_rcnn_model.pt', help='path to saved model')

    args = parser.parse_args()

    IMAGE_PATH = args.img
    MODEL_PATH = args.model

    image = cv2.imread(IMAGE_PATH)
    model = segmentation_model(MODEL_PATH)
    pred = model.detect_masks(image, rgb_image=False)   # rgb_image=False if loading image with cv2.imread()

    plotted = plot_masks(image,pred)
    
    os.makedirs('./results', exist_ok=True)
    cv2.imwrite('/results/res.jpg', plotted)

    plt.figure(figsize=(16,12))
    plt.imshow(plotted)
    plt.show()