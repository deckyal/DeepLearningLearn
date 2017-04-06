'''
Created on Apr 4, 2017

@author: deckyal
'''

import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg";
image = mpimg.imread(filename)
height,width,depth = image.shape

#create a Tensorflow variable. 
x = tf.Variable(image,name='x')
model = tf.global_variables_initializer()

with tf.Session() as session : 
    #x = tf.transpose(x, perm = [1,0,2])
    x = tf.reverse_sequence(x, np.ones((height,))*width,1,batch_dim = 0);
    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()

#