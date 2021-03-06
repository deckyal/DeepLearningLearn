'''
Created on May 25, 2017

@author: deckyal
'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

#Network parameters
n_input = 784 #Mnist img shape : 28*28
n_classes = 10
dropout = 0.75 #Probability to keep units

#TF graph input
x = tf.placeholder(tf.float32, [None,n_input])
y = tf.placeholder(tf.float32, [None,n_classes])
keep_prob = tf.placeholder(tf.float32)

#Create some wrappers for simplicity 
def conv2d(x,W,b,strides = 1):
    #Conv2D wrapper, with bias and relu activation 
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding = 'SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides = [1,k,k,1],padding = "SAME")

def conv_net(x,weights,biases,dropout):
    #Redhape input picture 
    x = tf.reshape(x, shape = [-1,28,28,1])
    
    #Convoluiton Layer
    conv1 = conv2d(x,weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k = 2)
    
    #Convoluiton Layer
    conv2 = conv2d(conv1,weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k = 2)
    
    #Fully connected layer 
    fc1 = tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
    print weights['wd1'].get_shape().as_list()
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    #Dropout
    fc1 = tf.nn.dropout(fc1,dropout)
    
    #Output class prediction 
    out = tf.add(tf.matmul(fc1,weights['out']), biases['out'])
    return out

#Store layers weight  and bias

weights = {
    #5x5 conv, 1 input, 32 output
    'wc1' : tf.Variable(tf.random_normal([5,5,1,32])),
    #5x5 conv, 32 input, 64 output 
    'wc2' : tf.Variable(tf.random_normal([5,5,32,64])),
    #Fully cnnected 7*7*64, 1024 outputs 
    'wd1' : tf.Variable(tf.random_normal([7*7*64,1024])),
    #1024 inputs, 10 outputs(class prediction) 
    'out' : tf.Variable(tf.random_normal([1024,n_classes]))
    }
    
biases = {
    'bc1' : tf.Variable(tf.random_normal([32])), 
    'bc2' : tf.Variable(tf.random_normal([64])), 
    'bd1' : tf.Variable(tf.random_normal([1024])), 
    'out' : tf.Variable(tf.random_normal([n_classes]))
    }

print tf.random_normal([32])

#construct model 
pred = conv_net(x,weights,biases, keep_prob)

#Define loss and optmizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Evaluate model 
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Initializing the variables 
init = tf.global_variables_initializer()

#Launch the graph 
with tf.Session() as sess : 
    sess.run(init)
    step = 1
    #Keep traiing until reach max iterations
    while step*batch_size < training_iters : 
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict  = {x:batch_x, y:batch_y, keep_prob: dropout})
        
        if step%display_step == 0 : 
            loss, acc = sess.run([cost,accuracy], feed_dict = {x: batch_x, y:batch_y, keep_prob : 1.})
            print "Iter"+str(step*batch_size) + ",Minitbatch loss = "+"{:.6f}".format(loss)+", Training accuraccy"+"{:.5f}".format(acc)
            
        step+=1
    print "Testing accuracy ", sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256], keep_prob : 1.})
        
    