import tensorflow as tf

def get_tfconfig():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    return config 

