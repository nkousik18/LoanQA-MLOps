import cv2
import numpy as np

def split_columns(img):
    """Split double-column pages for better reading order."""
    projection = np.sum(255 - img, axis=0)
    mid = img.shape[1] // 2
    for x in range(mid - 100, mid + 100):
        if projection[x] < np.mean(projection)*0.4:
            left, right = img[:, :x], img[:, x:]
            return [left, right]
    return [img]
