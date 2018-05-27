import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

def demo(images, titles, r, c, save_as=None):
    f, axis = plt.subplots(r, c, figsize=(24, 5 * r))
    axis = axis.ravel()

    for i, (image, title) in zip(range(0, len(axis)), zip(images, titles)):
        axis[i].imshow(image)
        axis[i].set_title(title, fontsize=15)

    if save_as:
        f.savefig(save_as)

    plt.show()

    
