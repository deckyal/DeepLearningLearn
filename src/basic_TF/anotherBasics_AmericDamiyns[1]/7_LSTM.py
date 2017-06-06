'''
Created on May 26, 2017

@author: deckyal
'''
import tensorflow as tf 
from tensorflow.contrib import rnn 
import numpy as np 

#Impor MNIST data 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#Parameters 
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

#Network Paramters 
n_input = 28 
n_steps = 28 
n_hidden = 128 
n_classes = 10 

#TF graph inpu t
x = tf.placeholder("float", [None, n_steps,n_input])
y = tf.placeholder("float", [None, n_classes])

#define weights 
weights = {
    'out' : tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }

biases = {
    'out' : tf.Variable(tf.random_normal([n_classes]))
    }

def RNN (x, weights, biases):
    x = tf.unstack(x,n_steps,1)
    
    #define a lstm cell with tensorlfow 
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    
    #get lstm cell output 
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
    
    #Linear activation, using rnn inner loop last output 
    return tf.matmul(outputs[-1], weights['out'])+biases['out']

pred = RNN(x, weights, biases)

#Define loss and optimnizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred , labels = y ))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#Evaluate model 
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))    
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Initializing the variables 
init = tf.global_variables_initializer()

with tf.Session() as sess : 
    sess.run(init)
    step= 1
    #Keep training utnil max iteraion 
    while step * batch_size  < training_iters : 
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        #Reshape ata to get 28 seq of 28 elements 
        batch_x = batch_x.reshape(batch_size, n_steps, n_input)
        #Run optimizationi op (backprop) 
        sess.run(optimizer, feed_dict = {x:batch_x, y : batch_y})
        #Caclulate the tacth accuracy 
        if step % display_step == 0 : 
            #Calculate the batch accuracy  
            acc = sess.run(accuracy, feed_dict = {x:batch_x, y:batch_y})
            #Calculate batch loss 
            loss = sess.run(cost, feed_dict = {x:batch_x, y:batch_y})
            print "iter " +str(step*batch_size) + "< Minitbatch loss = "+"{:.6f}".format(loss) + "Training accuracy " + "{:.5f}".format(acc)
        step+=1
    
    test_len = 128 
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label= mnist.test.labels[:test_len]
    print "Testing accruracy : ", sess.run(accuracy, feed_dict = {x:test_data, y:test_label})        
