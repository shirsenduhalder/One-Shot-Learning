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
Data directory paths

'''
root = '../data'
cub_root = '../data/cub/'
images_path = os.path.join(cub_root,'images/')
images_file = os.path.join(cub_root,'images.txt')
labels_file = os.path.join(cub_root,'image_class_labels.txt')
cache_folder = os.path.join(cub_root,'cache')
triplet_data_path = '../data/numpy_data/triplet_data.npy'
mean_features_path = '../data/numpy_data/mean_features.npy'
seen_labels_path = '../data/numpy_data/seen_labels.npy'
unseen_data_path = '../data/numpy_data/unseen_images.npy'
seen_data_path = '../data/numpy_data/seen_images.npy'
unseen_images_path = '../data/numpy_data/unseen_images.npy'


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
Only valid for training time. Divides the training dataset into seen and unseen classes
Returns: Seen and Unseen images along with their labels
'''
import random
def train_test_partition(images):
	#unique labels
	num_labels = np.unique(images[:,1])
	parition_ratio = 0.75

	train_images = []
	test_images = []

	for label in num_labels:
		total_label_images = np.array([image for image in images if image[1]==label])

		#determines the number of unseen classes
		number_label_test_images = int(parition_ratio * total_label_images.shape[0])
		
		#creates a random array of indices

		random_indices = random.sample(range(0,total_label_images.shape[0]),number_label_test_images)

		#chooses labels for unseen classes 
		label_train_images = [total_label_images[i] for i in random_indices]
		#chooses labels for seen classes
		label_test_images = [total_label_images[i] for i in range(total_label_images.shape[0]) if i not in random_indices]

		#unseen images array. Contains the image as well as their correspondng labels. Same goes for seen images array
		#label_train_images = np.ndarray.tolist(label_train_images)
		
		train_images.append(np.array(label_train_images))
		test_images.append(np.array(label_test_images))

	return np.array(train_images),np.array(test_images)


'''
Calculates the mean feature vecdtor of all labels
Returns: A ND array containing the mean feature vector along with the label
'''

def make_features_mean(images):
	#unique labels
	num_labels = images.shape[0]
	centroid_transfer_value = []

	#looping over al unique labels
	for i in range(num_labels):
		#getting images of the selected label
		features = images[i][:,0]
		#print(features.shape)

		#check feature shape if any issue
		label = images[i][0][1]

		#features of all images of the selected label
		#features = get_features(ims)

		#mean feature
		features_mean = np.mean(features,axis=0)
		
		#print('Mean features')

		#entry consisting of mean feature and it's corresponding label
		entry = []
		entry.append(features_mean)
		entry.append(label)
		centroid_transfer_value.append(entry)

	centroid_transfer_value = np.array(centroid_transfer_value)
	return centroid_transfer_value

def find_nearest_neighbours(compared_class, all_classes, n_neighbours):
	copies = [all_classes[compared_class]]*len(all_classes)
	all_classes = np.array(np.ndarray.tolist(all_classes))
	copies = np.array(copies)
	dist = np.linalg.norm(all_classes-copies, axis=1)
	indices = np.argsort(dist)
	return indices[1:n_neighbours+1]

def take_random_samples(class_data, n_datapoints=-1):
	random.shuffle(class_data)
	if n_datapoints == -1:
		return class_data
	return class_data[:n_datapoints]

def make_triplets_seen(seen_images, mean_features_seen):
	triplet_data = []

	for i in range(len(seen_images)):
		inp = []
		out_recons = []
		out_cross = []
		labels = []

		neighbour_list = find_nearest_neighbours(i, mean_features_seen[:,0], 5)

		temp_datapoints = []
		for n in neighbour_list:
			temp_datapoints.extend(take_random_samples(seen_images[n][:,0], 30))

		inp.extend(seen_images[i][:,0])
		out_recons.extend(seen_images[i][:,0])
		out_cross.extend(take_random_samples(temp_datapoints, seen_images[i].shape[0]))
		labels.extend(seen_images[i][:,1])

		triplet_data.append([inp, out_recons, out_cross, labels])
	return triplet_data

def make_triplets_unseen(seen_images, unseen_images, mean_features_seen, mean_features_unseen, mean_features):
	triplet_data = []
	
	for i in range(len(unseen_images)):
		inp = []
		out_recons = []
		out_cross = []
		labels = []

		neighbour_list = find_nearest_neighbours(i, mean_features[:len(mean_features_seen)+i][:,0], 5)
		print(neighbour_list)
		
		temp_datapoints = []
		for n in neighbour_list:
			if n < len(mean_features_seen):
				temp_datapoints.extend(take_random_samples(seen_images[n][:,0], 30))
			else:
				temp_datapoints.extend(take_random_samples(unseen_images[n-len(mean_features_seen)][:,0], 30))

		inp.extend(unseen_images[i][:,0])
		out_recons.extend(unseen_images[i][:,0])
		out_cross.extend(take_random_samples(temp_datapoints, unseen_images[i].shape[0]))
		labels.extend(unseen_images[i][:,1])

		triplet_data.append([inp, out_recons, out_cross, labels])
	return triplet_data

def make_triplets(seen_images, unseen_images, mean_features_seen, mean_features_unseen, mean_features):

	seen_triplet = make_triplets_seen(seen_images, mean_features_seen)
	unseen_triplet = make_triplets_unseen(seen_images, unseen_images, mean_features_seen, mean_features_unseen, mean_features)

	return np.array(seen_triplet), np.array(unseen_triplet)

def return_triplet_data():

	'''
	If data is not made before

	'''
	if not os.path.exists(triplet_data_path):
		'''
			Division of data according to mode
		'''
		##function is in data.py
		seen_images, unseen_images = seen_unseen_partition(images_path,images_file,labels_file,mode=mode,part_type=partition_type)
		#seen_features = get_features(seen_images[:,0])
		#unseen_features = get_features(unseen_images[:,0])
		#np.save(unseen_images_path,unseen_images)
		#print('Train images dim: {}'.format(seen_images.shape))
		
		##division of training images in seen and unseen
		train_images_seen,test_images_seen = train_test_partition(seen_images)
		train_images_unseen,test_images_unseen = train_test_partition(unseen_images)

		# np.save(train_images_seen_path,train_images_seen)
		# np.save(test_images_seen_path,test_images_seen)
		# np.save(train_images_unseen_path,train_images_unseen)
		# np.save(test_images_unseen_path,test_images_unseen)
		
		##unique seen labels
		#seen_labels = np.unique(train_images[:,1])

		# np.save(seen_labels_path,seen_labels)
		
		#print('Seen images dim: {}'.format(train_images.shape))
		#triplet_data = []

		'''
		Loading of mean features
		
		'''
		##checking if data exits
		#if not os.path.exists(mean_features_path):
			#making mean features
		mean_features_seen = make_features_mean(train_images_seen)
		mean_features_unseen = make_features_mean(train_images_unseen)
		mean_features = np.concatenate((mean_features_seen,mean_features_unseen),axis=0)
		#np.save(mean_features_path,mean_features)
		#else:
			#loading mean features
			#mean_features = np.load(mean_features_path)
		train_seen_triplets, train_unseen_triplets = make_triplets(train_images_seen, train_images_unseen, mean_features_seen, mean_features_unseen, mean_features)

	return train_seen_triplets, train_unseen_triplets, test_images_seen, test_images_unseen