import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess : 
    print "A : %i"% sess.run(a), "b : %i"%sess.run(b)
    print "Addition with constants : %i"%sess.run(a+b)
    print "Multplication with constants : %i"%sess.run(a*b)

#just declare the type, value will be inputted later    
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
c = tf.constant(4)

#Define the operation for placeholders
add = tf.add(a,b)
mul = tf.multiply(a,b)
#mul2 = tf.add(a,c)

#now we do the exact operations above here
with tf.Session() as sess : 
    print("Addition with variables : %i"%sess.run(add,feed_dict={a:2, b:3}))
    print('Multiplication with variables : %i'%sess.run(mul,feed_dict={a:2,b:3}))
    #print("Multpilication 2 : %i"%sess.run(mul2,feed_dict={a:2}))
    
matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess: 
    result = sess.run(product)
    print(result)