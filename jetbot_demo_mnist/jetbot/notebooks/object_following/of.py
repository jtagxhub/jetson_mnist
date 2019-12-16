

from jetcam.usb_camera import USBCamera
import cv2
import numpy as np

camera = USBCamera(capture_device=0, width=640, height=480, format='gray8')
img = camera.read()
thres = (np.amin(img) + np.amax(img)) / 2
(_, img) = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

