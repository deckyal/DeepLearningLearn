'''
Created on Apr 6, 2017

@author: deckyal
'''
import tensorflow as tf
import numpy as np 

x = tf.placeholder("float");
y= tf.placeholder("float");

w = tf.Variable([1.0,2.0],name = "w")
#model is y = a*x+break
y_model = tf.multiply(x,w[0]) + w[1]

#error is squareof the differences 
error = tf.square(y - y_model)

#The gradient desecent optimizer to minimeze the error 
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

#Normal tensorflow - initialize values, create a session and run the model 
model = tf.global_variables_initializer()

with tf.Session() as session : 
    session.run(model)
    for i in range(1000): 
        x_value = np.random.rand()
        y_value = x_value * 2 + 6
        session.run(train_op,feed_dict={x:x_value, y:y_value})
    
    w_value = session.run(w)
    print("Predicted model : {a:.3f}x + {b:.3f}".format(a=w_value[0],b=w_value[1]))