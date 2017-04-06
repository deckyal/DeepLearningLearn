'''
Created on Apr 4, 2017

@author: deckyal
'''
import tensorflow as tf

'''

a = tf.add(1,2,name="adding_1_by_two")
b = tf.multiply(a, 3,)
c = tf.add(4,5,"Another_addition")
d = tf.multiply(c,6,name="Multiplying")
e = tf.multiply(4,5)
f = tf.div(c, 6, )
g = tf.add(b,d)
h = tf.multiply(g,f)

with tf.Session() as sess: 
    #with is opening and alwasy ensure the file closed safely after the operation
    writer = tf.summary.FileWriter("/home/deckyal/workspace/DeepLearningLearn/src/basic_TF/output",sess.graph) 
    print(sess.run(h)) 
    writer.close()
    
'''

with tf.name_scope("MyOGroup") : 
    with tf.name_scope("ScopeA"): 
        a = tf.add(1,2,name="Add_these_numbers");
        b = tf.multiply(a,3);
    with tf.name_scope("ScopeB"): 
        c = tf.add(4,5,"Another_addition")
        d = tf.multiply(c,6,name="Multiplying")
    
    
with tf.name_scope("Scope_C") : 
    e = tf.multiply(4,5)
    f = tf.div(c, 6, )
    
g = tf.add(b,d)
h = tf.multiply(g,f)


with tf.Session() as sess: 
    #with is opening and alwasy ensure the file closed safely after the operation
    writer = tf.summary.FileWriter("/home/deckyal/workspace/DeepLearningLearn/src/basic_TF/output",sess.graph) 
    print(sess.run(h)) 
    writer.close()