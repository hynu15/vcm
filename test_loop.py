import cv2
import numpy as np

img = np.zeros((100, 100, 3), dtype=np.uint8)
roi = np.ones((100, 100), dtype=np.uint8)
roi_3 = np.repeat(roi[..., None], 3, axis=2)
blurred = cv2.GaussianBlur(img, (51, 51), 20)
hybrid = np.where(roi_3 == 1, img, blurred).astype(np.uint8)

cv2.imwrite("test_img.png", hybrid)
