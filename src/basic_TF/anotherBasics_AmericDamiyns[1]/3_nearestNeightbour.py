'''
Created on May 25, 2017

@author: deckyal
'''

import numpy as np 
import tensorflow as tf 

#import MNIST data
from tensorflow.examples.tutorials.mnist import input_data 
#we use input_data.py to download and preprocess data in tf format
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

xtr = tf.placeholder("float", [None,784])
xte = tf.placeholder("float", [784])

distance = tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))), reduction_indices = 1)
#Prediction: get min distance index (nearest neighbour)
pred = tf.arg_min(distance,0)

accuracy = 0
#initializing all variables 
print Xte
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
    
