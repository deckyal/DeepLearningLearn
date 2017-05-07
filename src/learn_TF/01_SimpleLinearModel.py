'''
Created on May 4, 2017

@author: deckyal
'''

import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
from sklearn.metrics import confusion_matrix 
from tensorflow.core.framework import cost_graph_pb2


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/",one_hot = True)

print("Size of : ")
print("Treining set : \t {}".format(len(data.train.labels)))
print("Test set : \t {}".format(len(data.train.labels)))
print("Validation set : \t {}".format(len(data.train.labels)))

#We map the class to be single integer 
data.test.cls = np.array([label.argmax() for label in data.test.labels])

print(data.test.cls[0:5])


#function for plotting images 

def plot_images(images,cls_true,cls_pred = None):
    assert len(images) == len(cls_true) == 9
    
    #Create figure with 3x3 sub-plots 
    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace = 0.3)
    
    for i,ax in enumerate(axes.flat) : 
        #plot the image 
        ax.imshow(images[i].reshape(img_shape),cmap = 'binary')
        
        #Show true and predicted classes . 
        if cls_pred is None : 
            xLabel = "True : {0}".format(cls_true[i])
        else : 
            xLabel = "True : {0}, Pred : {1}".format(cls_true[i],cls_pred[i])
        
        ax.set_xlabel(xLabel)
        
        #Remove ticks from the plot 
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
    

if __name__ == '__main__':
    
    img_size = 28
    img_size_flat = img_size * img_size
    img_shape = (img_size, img_size)
    
    num_classes = 10
    
    #Get the first images from the test set
    images = data.test.images[0:9]
    
    #Get the true classes for those images 
    cls_true = data.test.cls[0:9]
    
    #Plot the images and labels using our helper-function above. ; 
    plot_images(images=images, cls_true = cls_true)
    
    ## Now making the placeholder variables 
    x = tf.placeholder(tf.float32, [None,img_size_flat])
    y_true = tf.placeholder(tf.float32,[None,num_classes])
    y_true_cls = tf.placeholder(tf.int64,[None])
    
    #Variable to be optimized 
    weights = tf.Variable(tf.zeros([img_size_flat,num_classes]))
    biases = tf.Variable(tf.zeros([num_classes]))
    
    #Make the model 
    logits = tf.matmul(x,weights) + biases 
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.arg_max(y_pred,dimension = 1)
    
    #Cost function for optimization 
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_true)
    
    cost = tf.reduce_mean(cross_entropy)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost)
    
    #Performance measurement 
    correct_prediction = tf.equal(y_pred_cls,y_true_cls)
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    #TensorFlow Run 
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    batch_size = 100
    
    feed_dict_test = {x:data.test.images,y_true : data.test.labels, y_true_cls : data.test.cls}
    
    #Now optimization !
    num_iterations  = 1; 
    for i in range(num_iterations) : 
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        
        #Put the batch into a dict with the proper names 
        feed_dict_train = {x:x_batch, y_true: y_true_batch }
        session.run(optimizer,feed_dict = feed_dict_train)

    #Now printing the accuracy 
    
    feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}
    
    #Use Tensorflow to compute the accuracy 
    acc = session.run(accuracy,feed_dict = feed_dict_test)
    
    #Print the accuracy 
    print("Accuracy on test-set: {0:.1%}".format(acc))
    
    #Printing the confusion matrix 
    cls_true = data.test.cls
    cls_pred = session.run(y_pred_cls,feed_dict = feed_dict_test)
    
    cm = confusion_matrix(y_true = cls_true, y_pred = cls_pred)
    
    print(cm)
    
    #Plot the cm as an image 
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)
    
    #Little adjustments
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks,range(num_classes))
    plt.yticks(tick_marks,range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    #Plotting some example errors
    
    correct,cls_pred = session.run([correct_prediction,y_pred_cls],feed_dict=feed_dict_test)
    
    
    #Negate the boolean array. 
    incorrect = (correct == False)
    
    images = data.test.images[incorrect]
    
    cls_pred = cls_pred[incorrect]
    
    cls_true = data.test.cls[incorrect]
    
    plot_images(images = images[0:9],cls_true = cls_true[0:9],cls_pred = cls_pred[0:9])
    
    #Plotting weights 
    
    w = session.run(weights)
    
    w_min = np.min(w)
    w_max = np.max(w)
    
    fig,axes = plt.subplots(3,4)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    
    for i,ax in enumerate(axes.flat) : 
        if i<10: 
            image = w[:,i].reshape(img_shape)
            ax.set_xlabel("Weights : {0}".format(i))
            ax.imshow(image,vmin=w_min,vmax = w_max, cmap='seismic')
            
        ax.set_xticks([])
        ax.set_yticks([])
    
    
    
               
               