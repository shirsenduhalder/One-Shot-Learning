import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
import time
import os 
from datetime import timedelta
from mlxtend.preprocessing import one_hot
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from data import *
from triplet import *

def make_one_hot(labels):
	unique_classes = np.unique(labels)
	mapper= {}
	i = 0
	for c in unique_classes:
		mapper[c] = i
		i += 1
	new_labels = [mapper[x] for x in labels]
	new_labels_oh = one_hot(new_labels)
	return mapper, new_labels_oh

triplet_data, seen_labels, unseen_images, seen_images = return_triplet_data()

label_mapper, triplet_labels = make_one_hot(triplet_data[:,0,1])


latent_dim1 = 450
data_dim = 2048
label_dim = triplet_labels.shape[1]
#placeholders

X11 = tf.placeholder(tf.float32, shape=[None,data_dim], name='Seen_vis')
X12 = tf.placeholder(tf.float32, shape=[None,data_dim], name='Seen__cross_vis')
X13 = tf.placeholder(tf.float32, shape=[None, data_dim], name='Seen_cross_class')

Y1 = tf.placeholder(tf.float32, shape=[None, label_dim], name='Seen_label')
#Weights and biases

W11 = tf.Variable(tf.random_normal([data_dim,latent_dim1]), name='W11')
W31 = tf.Variable(tf.random_normal([latent_dim1, label_dim]), name='W31')

bias11 = tf.Variable(tf.random_normal([latent_dim1]), name='bias11')
bias12 = tf.Variable(tf.random_normal([data_dim]), name='bias12')
bias31 = tf.Variable(tf.random_normal([label_dim]), name='bias31')

# In[185]:

with tf.name_scope('latent_space'):
    latent_seen1 = tf.nn.relu(tf.add(tf.matmul(X11,W11),bias11), name = 'latent_seen1')

with tf.name_scope('seen_outputs'):
    rec_seen = tf.nn.relu(tf.add(tf.matmul(latent_seen1,tf.transpose(W11)),bias12), name = 'auto_seen')
    rec_cross_class = tf.nn.relu(tf.add(tf.matmul(latent_seen1, tf.transpose(W11)), bias12), name = 'source_cross_seen')


with tf.name_scope('classifier'):
    logit  = tf.add(tf.matmul(latent_seen1,W31),bias31, name = 'vis_class')
    class_loss = tf.nn.softmax_cross_entropy_with_logits(logits = logit, labels = Y1)

# In[186]:

#total loss

classifier_loss_t = 1000*tf.reduce_mean(class_loss) + tf.norm(W31) + tf.norm(bias31)

source_loss = tf.norm(tf.subtract(rec_seen, X12))  - tf.norm(tf.subtract(rec_cross_class, X13)) + tf.norm(W11) + tf.norm(bias11) + tf.norm(bias12)

total_loss = classifier_loss_t + source_loss
                                                                                             
    
#training classification accuracy on seen samples

acc = tf.equal(tf.argmax(tf.nn.softmax(logit), 1), tf.argmax(Y1, 1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# In[187]:


optimizer=tf.train.AdamOptimizer(0.01).minimize(total_loss)
init=tf.global_variables_initializer()


# In[188]:

def get_batches_seen(triplet_data, triplet_labels, total_batch,name):

        batches1 = []
        batches2 = []
        batches3 = []
        label = []

        if total_batch == 0:
            return [triplet_data[:,0,0], triplet_data[:,1,0], triplet_data[:,2,0]. triplet_labels]
        
        for i in range(total_batch):
       	    temp_batch = [x[0] for x in triplet_data[i*batch_size:(i+1)*batch_size, 0, 0]]
            batches1.append(temp_batch)

            temp_batch = [x[0] for x in triplet_data[i*batch_size:(i+1)*batch_size, 1, 0]]
            batches2.append(temp_batch)

            temp_batch = [x[0] for x in triplet_data[i*batch_size:(i+1)*batch_size, 2, 0]]
            batches3.append(temp_batch)

            label.append(triplet_labels[i*batch_size:(i+1)*batch_size, :])
        
        return np.array(batches1), np.array(batches2), np.array(batches3), np.array(label)



def get_batches_train_unseen(triplet_data,seen_labels,total_batch,name):

	pass



with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    
    sess.run(init)
    batch_size = 50
    total_batch = int(triplet_data.shape[0]/batch_size)
    
    batches_seen1, batches_seen2, batches_seen3, batches_seen_label  = get_batches_seen(triplet_data, triplet_labels, total_batch, 'train') 


    for epoch in range(35000):

        avg_cost = 0
        avg_acc = 0
        avg_acc1 = 0
        
        for i in range(total_batch):
            batch_x11 = batches_seen1[i].astype(np.float32)
            batch_x12 = batches_seen2[i].astype(np.float32)
            batches_x13 = batches_seen3[i].astype(np.float32)
            batch_x1_label = batches_seen_label[i].astype(np.float32)
           
            _ , c, acc_new = sess.run([optimizer, total_loss, acc], feed_dict = {X11: batch_x11, X12: batch_x12, X13: batches_x13, Y1: batch_x1_label})
                    
            avg_cost += c /total_batch
            avg_acc += acc_new /total_batch

        
        #logit2, acc_t = sess.run([logit2, acc2], {X2: features_t, Y2: label_t})

           
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost), "accuracy =", "{:.5f}".format(avg_acc))
     
    
    print ("Training of seen classes complete!")


    #batches_seen,batches_seen_label,batches_unseen, batches_unseen_label = get_batches_train_unseen(triplet_data,triplet_labels,total_batch,name)

    for epoch in range(35000):
    	pass

    sess.close()