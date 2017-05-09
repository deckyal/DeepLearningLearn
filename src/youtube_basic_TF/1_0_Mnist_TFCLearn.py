'''
Created on May 5, 2017

@author: deckyal
'''
import tensorflow as tf
import numpy as np 

if __name__ == '__main__':
    
    #Declaring list of features 
    features = [tf.contrib.layers.real_valued_column("x",dimension = 1)]
    
    #make the estimator 
    estimator = tf.contrib.learn.LinearRegressor(feature_columns = features)
    
    x = np.array([1.,2.,3.,4.])
    y = np.array([0.,-1.,-2.,-3.])
    
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x},y,batch_size = 4, num_epochs = 1000)
    
    estimator.fit(input_fn = input_fn, steps = 1000)
    
    print(estimator.evaluate(input_fn = input_fn))