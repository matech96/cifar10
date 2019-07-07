import PIL
import numpy as np
from scipy.misc import imresize


def open_image_np(image_file_name, height, width):
    img = PIL.Image.open(image_file_name).convert('RGB')
    img = img.resize((height, width))
    img = np.asarray(img, dtype=np.uint8).astype(np.float32)
    if len(img.shape) == 2:
        img.resize((height, width, 1))
        img = np.repeat(img, 3, axis=-1)
    return img


def open_image_batch(image_file_names, height, width):
    current_batch_size = len(image_file_names)
    images = np.zeros((current_batch_size, height, width, 3))
    for image_id, image_file_name in enumerate(image_file_names):
        images[image_id, :] = open_image_np(image_file_name, height, width)
    return images
