'''
Created on May 15, 2017

@author: deckyal
'''
#Create two variable.s 

import tensorflow as tf

weights = tf.Variable(tf.random_normal([784,200],stdder = 0.35), name ="Weights")
biases = tf.Variable(tf.zeros([200]), name = "biases")

#Pinggin variable 
with tf.device("/cpu:0"): 
    v = tf.Variable(tf.zeros([200]), name = "bCpu0")
    
with tf.device("/gpu:0"): 
    v = tf.Variable(tf.zeros([200]), name = "bGpu0")

with tf.device("/job:ps/task:7") : 
    v = tf.Variable(tf.zeros([200]), name = "bTask7")
    

#Create another variable with the samve value as weithgs 
w2 = tf.Variable(weights.initialized_value(), name = "w2")

#Create some variables. 
v1 = tf.Variable(...,name = "V1")
v2 = tf.Variable(...,name = "V2")

    
#Add an op to initialize the variables 
init_op = tf.global_variables_initializer()

saver = tf.train.Saver(0)
#to save v2 in other name 
#saver = tf.train.Saver({"mv_v2":v2})

#Later launch the model, initialize the variables, do some works, save the 
#variables to disk. 

with tf.Session() as sess : 
    sess.run(init_op)
    #do some work with the model
     
    #save the variables to disk 
    save_path = saver.save(sess,"/tmp/model.ckpt")
    print("Model saved to  : %s"%save_path)
    
    
#to restore 
with tf.Sesssion() as sess: 
    #restore the variables 
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored")