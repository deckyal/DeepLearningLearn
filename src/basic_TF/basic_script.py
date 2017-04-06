'''
Created on Mar 30, 2017

@author: deckyal
'''

x = 35
y = x+5
print(y)

import tensorflow as tf 

#x = tf.constant(35,name='x')
#x = tf.constant([35,40,45],name = 'x')
#y = tf.Variable(x+5,name = 'y')
#x = tf.constant([35,40,45], name = 'x')

#We just created them, but didn't execute them yet. Kind of declaration 
#print y

'''
import numpy as np 
data = np.random.randint(1000,size = 10000)

y = tf.Variable((5*data**2)-3*data-15,name = 'y')


#Now initialized the variables. 

model = tf.global_variables_initializer()

with tf.Session() as session : 
    #now create the sessin and run all declaration along the process on the 'model' data .
    session.run(model)
    print session.run(y) #print the value of y here
'''

'''
x = tf.Variable(0,name ='x')

model = tf.global_variables_initializer()

with tf.Session() as session : 
    for i in range(5) : 
        session.run(model)
        x = x+1
        print (session.run(x))
'''

'''

x = tf.constant(35,name = 'x')
print(x)

y = tf.Variable(x + 5, name = 'y')

with tf.Session() as sessionx: 
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/basic',sessionx.graph)
    model = tf.global_variables_initializer()
    sessionx.run(model)
    print(sessionx.run(y))
    
'''
        
        
import numpy as np 

with tf.Session() as session : 
    for i in range(5) :  
        
        data = np.random.randint(1000,size = 10000)
        t_mean = tf.Variable(np.average(data),name = 't_mean')
        
        model = tf.global_variables_initializer()
        
        session.run(model) #always run the model
        print (session.run(t_mean)) #print it by runing it in tensorflow
