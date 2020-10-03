import cv2
import numpy as np
import matplotlib.pyplot as plt
from mask_rcnn import *

IMAGE_PATH = 'test_image.jpg'
MODEL_PATH = './maskrcnn_saved_models/mask_rcnn_model_epoch_36.pt'

image = cv2.imread(IMAGE_PATH)
model = segmentation_model(MODEL_PATH)
pred = model.detect_masks(image, rgb_image=False)   # rgb_image=False if loading image with cv2.imread()

plotted = plot_masks(image,pred)
plt.figure(figsize=(16,12))
plt.imshow(plotted)