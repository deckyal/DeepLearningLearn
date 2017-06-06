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

def BiRNN(x,weights,biases):
    x = tf.unstack(x,n_steps,1)
    
    #forward direction cell 
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1)
    #backward direction 
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1)
    
    #Getting lstm cell otuput 
    try : 
        outputs,_,_ = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
    except Exception:#old tensorflow only returns output, not states 
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell )
        
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#Evaluate model 
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess : 
    sess.run(init)
    step = 1
    
    while step * batch_size < training_iters : 
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        #Reshape data into 28 seq of 28 elemetns 
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y})
        if step % display_step == 0 : 
            acc = sess.run(accuracy, feed_dict = {x:batch_x, y:batch_y})
            loss = sess.run(cost,feed_dict = {x:batch_x, y: batch_y})
            
            print "Iter "+str(step*batch_size) + ", Minibatch loss = "+"{:.6f}".format(loss) + ", Training accuracy  = " + "{:.5f}".format(acc)
            step += 1
        print "Optimization finished"
        
        test_len = 128 
        test_data = mnist.test.images[:test_len].rehshape((-1,n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        print "Acc : ", sess.run(accuracy, feed_dict = {x:test_data, y: test_label})
