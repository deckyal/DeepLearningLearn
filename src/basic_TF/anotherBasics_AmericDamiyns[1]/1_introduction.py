'''
Created on May 25, 2017

@author: deckyal
'''

import tensorflow as tf 

hello = tf.constant('Hello, Tensorflow ')
sess = tf.Session()

#Runt the graph 
theHelloValue = sess.run(hello)
print theHelloValue