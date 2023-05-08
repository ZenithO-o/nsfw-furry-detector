# This is for my own sanity, I did not set up TF properly on my PC and I'm too lazy to fix it
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import cv2
import tensorflow as tf

TFLITE_FILE = './nsfw_model.tflite'
IMAGE_SIZE  = (384, 384)

def img_path_to_input(image_path: str) -> np.ndarray:
    """Loads and applies all the necessary transforms to the image for it to be
    ready to inference with the TFLite model.

    Args:
        image_path (str): path to image

    Returns:
        np.ndarray: normalized image ready for inference
    """
    # load image
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # preprocessing (resize + normalization between [0, 1 + float32)
    im = cv2.resize(im, IMAGE_SIZE) / 255
    reshape_size = [1]+list(im.shape)
    im = np.reshape(im, reshape_size)
    im = im.astype(np.float32)
    
    return im

def run_interpreter(input: np.ndarray, interpreter: tf.lite.Interpreter) -> int:
    """Runs TFLite interpreter for NSFW model on an image.

    Args:
        input (np.ndarray): input image ready for inference
        interpreter (tf.lite.Interpreter): loaded TFLite interpreter model 

    Returns:
        int: 0 if NSFW, 1 if SFW
    """
    # Get index of input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return np.argmax(output_data[0])


if __name__ == '__main__':
    interpreter = tf.lite.Interpreter(TFLITE_FILE)
    interpreter.allocate_tensors()

    model_input = img_path_to_input('test_image.jpg')
    model_output = run_interpreter(model_input, interpreter)
    print(model_output)
    