import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
import time
import os 
from datetime import timedelta
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data import *
import inception

from inception import transfer_values_cache

'''
Directory paths

'''
root = '../data/cub/'
images_path = os.path.join(root,'images/')
images_file = os.path.join(root,'images.txt')
labels_file = os.path.join(root,'image_class_labels.txt')
cache_folder = os.path.join(root,'cache')

'''
Dataset generation parameters

'''

partition_type = 'linear' #, 'random'
mode = 'train_mode'#,'test_mode'
neighbour_number = 5
'''
Cache for data
'''
file_path_cache_train = os.path.join(cache_folder,'inception_cub_train.pkl')
file_path_cache_test = os.path.join(cache_folder,'inception_cub_test.pkl')

#Loading model for the feature values
model = inception.Inception()


'''
Returns feature values of a numpy array
WARNING: Array has to be 4D
Outputs a N X 2048 feature vector. N are the number of total items
'''

def get_features(images):
	#Ineption Net uses values of images from (0,255). Hence the checkpoint ensures the normalization correctness
	if(np.max(images[0])<=1.0):
		images = images * 255.0

	#returning features values
	features = transfer_values_cache(images=images,model=model)

	return features

'''
Division of data according to mode
'''

if(mode == 'train_mode'):
	#function is in data.py
	train_images = seen_unseen_partition(images_path,images_file,labels_file,mode=mode,part_type=partition_type)
else:
	test_images = seen_unseen_partition(images_path,images_file,labels_file,mode=mode,part_type=partition_type)


'''
Only valid for training time. Divides the training dataset into seen and unseen classes
Returns: Seen and Unseen images along with their labels
'''

def make_seen_unseen(train_images):
	#unique labels
	num_labels = np.unique(train_images[:,1])
	
	#determines the number of unseen classes
	parition_ratio = 0.25
	unseen_classes_number = int(parition_ratio * num_labels.shape[0])
	print('Number of unseen classes: {}'.format(unseen_classes_number))
	
	#creates a random array of indices
	random_indices = np.random.randint(0,num_labels.shape[0],unseen_classes_number)

	#chooses labels for unseen classes 
	unseen_labels = np.take(num_labels, random_indices)

	#chooses labels for seen classes
	seen_labels = np.setdiff1d(num_labels,unseen_labels)

	#unseen images array. Contains the image as well as their correspondng labels. Same goes for seen images array
	unseen = np.array([image for image in train_images for label in unseen_labels if image[1] == label])
	seen = np.array([image for image in train_images for label in seen_labels if image[1] == label])

	return seen,unseen

'''
Calculates the mean feature vecdtor of all labels
Returns: A ND array containing the mean feature vector along with the label
'''

def make_features_mean(train_images):
	#unique labels
	num_labels = np.unique(train_images[:,1])
	centroid_transfer_value = []

	#looping over al unique labels
	for i in range(num_labels.shape[0]):
		#getting images of the selected label
		label_images = train_images[train_images[:,1] == num_labels[i]]
		label_images = label_images[:,0]

		#features of all images of the selected label
		features = get_features(label_images)

		#mean feature
		features_mean = np.mean(features,axis=0)
		
		#entry consisting of mean feature and it's corresponding label
		entry = []
		entry.append(features_mean)
		entry.append(num_labels[i])
		centroid_transfer_value.append(entry)

	centroid_transfer_value = np.array(centroid_transfer_value)
	return centroid_transfer_value


def return_c2_im(train_images,query_im_label,neighbour_number):
	
	#randomly selected label L1 and query image
	selected_label = query_im_label[1]
	selected_image = query_im_label[0]
	print('Selected label: {}'.format(selected_label))
	
	#feature of the query image
	selected_features = get_features(selected_image[np.newaxis,:,:,:])

	unique_labels = np.unique(train_images[:,1])

	#mean feature values of all other labels L
	mean_features = make_features_mean(train_images)

	diff_score = []
	for i in range(unique_labels.shape[0]):
		#same label L1 should not be considered
		if(mean_features[i,1] == selected_label):
			diff = float('inf')
			entry = [diff,unique_labels[i]]
			diff_score.append(entry)
		else:
			#Euclidean norm
			diff = np.linalg.norm(mean_features[i,0] - selected_features)
			entry = [diff,unique_labels[i]]
			diff_score.append(entry)

	#sorting according to the ascending order of L2 norm
	sorted_scores = np.array(sorted(diff_score,key = lambda entry:entry[0]))

	#selecting only labels
	nearest_labels = sorted_scores[:,1][:neighbour_number]
	np.random.shuffle(nearest_labels)
	print('Nearest label:{}'.format(nearest_labels[0]))
	
	#all images of a randomly selected Class label L2
	nearest_label_images = np.array([image[0] for image in train_images if image[1] == nearest_labels[0]])

	#selecting a random image from L2
	np.random.shuffle(nearest_label_images)

	return np.expand_dims(nearest_label_images[0],axis=0),sorted_scores

# def make_triplet(images_path,images_file,labels_file,mode,neighbour_number):
neighbour_number = 5

#unique labels
labels = np.unique(train_images[:,1])

#selecting a random label L1
np.random.shuffle(labels)
selected_label = labels[0]

print('Selected_label:{}'.format(selected_label))

#images of the selected label L1
selected_label_images = np.array([image for image in train_images if image[1] == selected_label])

#This does not allow same image from a same label L1 as c1_im and query_im
rand1 = np.random.randint(0,selected_label_images.shape[0])
rand2 = np.random.randint(0,selected_label_images.shape[0])
while(rand1==rand2):
	rand2 = np.random.randint(0,selected_label_images.shape[0])

query_im_and_label = selected_label_images[rand1]
c1_im_and_label = selected_label_images[rand1]
c2_im_and_label,sorted_scores = return_c2_im(train_images,query_im_and_label,neighbour_number)