'''
Created on Apr 5, 2017

@author: deckyal
'''

import tensorflow as tf 
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/olympic2016.csv"

features = tf.placeholder(tf.int32, shape=[3], 'features')
country = tf.placeholder(tf.string,name = 'country')
total = tf.reduce_sum(features,name = 'total')


printerop = tf.Print(total,[country,features,total], name = 'printer')

