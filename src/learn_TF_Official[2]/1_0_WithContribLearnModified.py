'''
Created on May 7, 2017

@author: deckyal
'''

import numpy as np
import tensorflow as tf 

#list of features 
def model(features, labels, mode):
    W = tf.get_variable("W", [1], tf.float64)
    b = tf.get_variable("b",[1],tf.float64)
    y = W*features['x'] + b
    
    loss = tf.reduce_sum(tf.square(y - labels))
    
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step,1))
    
    return tf.contrib.learn.ModelFnOps(mode = mode,predictions = y, loss = loss, train_op = train)

estimator = tf.contrib.learn.Estimator(model_fn = model )

#Our data sets 
x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1.,-2.,-3.])

input_fn =  tf.contrib.learn.io.numpy_input_fn({"x":x},y,4,1000)
estimator.fit(input_fn=input_fn, steps = 1000)
print(estimator.evaluate(input_fn = input_fn, steps = 10))

