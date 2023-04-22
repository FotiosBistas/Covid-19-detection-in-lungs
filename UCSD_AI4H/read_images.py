import os
import tensorflow as tf 
import numpy as np  

#this might need adjusting 
IMG_HEIGHT = 224 
IMG_WIDTH  = 224 

def readImageNames(filename): 
    with open(filename) as f: 
        lines = f.read().splitlines() 
        print("Read lines from file: " + filename)
    return lines 



def loadImage(filename): 

    print("Reading image: " + filename)
    img = tf.io.read_file(filename)
    
    _ ,file_extension = os.path.splitext(filename)

    if file_extension == '.png': 
        img = tf.image.decode_png(img, channels=1)
    elif file_extension == '.jpg':
        img = tf.image.decode_jpeg(img, channels=1)
    else: 
        print("Couldn't decode")
        return
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0 
    return img  




