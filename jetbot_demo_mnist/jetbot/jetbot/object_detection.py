import tensorrt as trt
from .tensorrt_model import TRTModel
import numpy as np
import cv2

def bgr8_to_ssd_input(camera_value):
    #x = cv2.imread("/home/jetbot/6.png")
    #x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = 1.0 - camera_value / 255.0
    x = x.reshape(1,28,28)
    return x


class ObjectDetector(object):
    
    def __init__(self, engine_path, preprocess_fn=bgr8_to_ssd_input):
        logger = trt.Logger()
        self.trt_model = TRTModel(engine_path, input_names=['in'],
                                  output_names=['out'])
        self.preprocess_fn = preprocess_fn
        
    def execute(self, *inputs):
        trt_outputs = self.trt_model(self.preprocess_fn(*inputs))
        return trt_outputs
    
    def __call__(self, *inputs):
        return self.execute(*inputs)
