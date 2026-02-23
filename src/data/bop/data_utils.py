import numpy as np


def idx2coords(choose, img_size):
    h, w = img_size
    x = choose % w
    y = choose // w
    coords = np.stack([x, y], axis=1)
    return coords.astype(np.float32)
