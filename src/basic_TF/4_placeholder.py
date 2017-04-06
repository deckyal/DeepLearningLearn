'''
Created on Apr 5, 2017

@author: deckyal
'''
import tensorflow as tf

x = tf.placeholder("float", [None,3],) #x as placeholder don't have data yet. 
y = x*2

with tf.Session() as session : 
    x_data = [[1,2,3],[4,5,6]]
    result = session.run(y,feed_dict={x:x_data}) #We feeed to placeholder now. 
    print result;
    
    
    
#