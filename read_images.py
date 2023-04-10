import os
import tensorflow as tf 
import numpy as np  


IMG_HEIGHT = 224 
IMG_WIDTH  = 224 
BATCH_SIZE = 32 

def readImageNames(filename): 
    with open(filename) as f: 
        lines = f.readlines() 
        print("Read lines from file: " + filename)
    return lines 



def loadImage(filename, label): 
    _ , extension = os.path.splitext(filename)
    img = tf.io.read_file(filename)
    if extension == ".png":
        img = tf.image.decode_png(img, channels=1) 
    else: 
        img = tf.image.decode_jpeg(img, channels=1)