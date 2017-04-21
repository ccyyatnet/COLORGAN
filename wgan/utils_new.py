import numpy as np
from scipy import misc
#import matplotlib.pyplot as plt

def cvtRGB2YUV(image):
    cvt_matrix = np.array([[0.299, -0.169, 0.5],
                                                [0.587, -0.331, -0.419],
                                                [0.114, 0.5, -0.081]], dtype = np.float32)
    return image.dot(cvt_matrix) + [0, 127.5, 127.5]

def cvtYUV2RGB(image):
    cvt_matrix = np.array([[1, 1, 1],
                                                [-0.00093, -0.3437, 1.77216],
                                                [1.401687, -0.71417, 0.00099]],dtype = np.float32)
    return (image - [0, 127.5, 127.5]).dot(cvt_matrix).clip(min=0,max=255)

def transform(image, center_crop_size=64, is_crop=True, resize_w=64, color_space = "RGB"): #with resize
    if is_crop:
        cropped_image = center_crop(image, center_crop_size, resize_w=resize_w)
    else:
        cropped_image = image
    if color_space == "YUV":
        cropped_image = cvtRGB2YUV(cropped_image)
    return np.array(cropped_image)/127.5 - 1. 

def save_image(image, image_path, color_space = "RGB"):
    image = (image+1.)*127.5
    if color_space == "YUV":
        image = cvtYUV2RGB(image)
    return misc.imsave(image_path, image)

def save_images(images, size, image_path, color_space = "RGB"):
    images = (images+1.)*127.5
    merged_image = merge(images, size)
    if color_space == "YUV":
        merged_image = cvtYUV2RGB(merged_image)
    return misc.imsave(image_path, merged_image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def show_image(image, image_name, window_idx, close = True):
    plt.figure(window_idx)
    plt.imshow(image)
    plt.axis('off')
    plt.title(image_name)
    plt.show()
    if close:
      plt.close(window_idx)

def show_images(images, size, window_idx, close=True):
    plt.figure(window_idx)
    merged_image = cvtYUV2RGB(merge(images, size))
    plt.imshow(merged_image)
    plt.axis('off')
    plt.show()
    if close:
      plt.close(window_idx)