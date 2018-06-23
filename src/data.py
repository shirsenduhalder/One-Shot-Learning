import matplotlib.pyplot as plt
import numpy as np 
import os 
import re
from natsort import natsorted
from skimage.transform import resize
import sys
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

def file_names(images_file,labels_file):

	image_names = open(images_file,'r')
	image_names = image_names.read()
	image_names = re.split('\n',image_names)
	
	#sorted according to characters rather than binary
	image_names = natsorted(image_names)
	
	#because of bug in .txt file
	image_names.pop(0)

	image_details = []

	for line in image_names:
		det = line.split(' ')
		#making a detail dict
		details = {}
		details['index'] = det[0]
		details['name'] = det[1]
		image_details.append(details)

	image_labels = open(labels_file,'r')
	image_labels = image_labels.read()
	image_labels = re.split('\n',image_labels)

	#sorted according to characters rather than binary
	image_labels = natsorted(image_labels)

	for i in range(len(image_details)):
		im_no = image_details[i]['index']
		label_im_no = image_labels[i].split(' ')[0]
		if(int(im_no) == int(label_im_no)):
			image_details[i]['class_label'] = image_labels[i].split(' ')[1]

	return image_details

def return_images(images_path,images_file,labels_file):

	images = []
	image_details = file_names(images_file,labels_file)
	count = 0
	for i in range(len(image_details[:1000])):
		#dynamic printing on terminal
		sys.stdout.write('Loaded {}/{} images \r'.format(i,len(image_details)))
		sys.stdout.flush()

		#image dict
		image = {}
		np_image = plt.imread(os.path.join(images_path,image_details[i]['name']))
		
		#checkpoint for BW images
		if(np_image.ndim == 2):
			np_image = np_image[:,:,np.newaxis]
			np_image = np.tile(np_image,(1,1,3))

		#resizing image
		np_image = resize(np_image,(300,300),mode='reflect')
		np_image = np_image.reshape(-1,300,300,3)
		#print(np_image.shape)
		#typecasting
		image_feature = get_features(np_image)
		image_feature = image_feature.reshape(2048)
		#adding image and label to the image dictionary
		image['feature'] = image_feature
		image['label'] = image_details[i]['class_label']

		#adding single image to the list of images
		images.append(image)
	
	return np.array(images)



def seen_unseen_partition(images_path,images_file,labels_file,mode,part_type='linear'):

	images = return_images(images_path,images_file,labels_file)
	
	#linear partition
	if(part_type == 'linear'):

		#total number of labels
		num_classes = int(images[-1]['label'])
		
		#number of classes in the unseen dataset
		num_classes_unseen = int(np.ceil(0.25*num_classes))
		
		#starting index of the partition label
		partition_label = num_classes - num_classes_unseen + 1

		start_idx = 0
		for i in range(images.shape[0]):
			if(int(images[i]['label']) == partition_label):
				start_idx = i

				#Break on reaching partition index
				break
		
		#seen dict till partition index
		seen_dict = images[:start_idx]

		#unseen dict after partition index
		unseen_dict = images[start_idx:]

		#clearing memory
		del images

	#random partition
	if(part_type=='random'):

		seen_dict = []
		unseen_dict = []

		#number of classes
		num_classes = int(images[-1]['label'])

		#number of classes in unseen set
		num_classes_unseen = int(np.ceil(0.25*num_classes))
		
		#last label number
		last_label = int(images[-1]['label'])
		
		#randomly selecting unseen label indices
		unseen_labels = np.random.randint(1,last_label,num_classes_unseen)

		for i in range(images.shape[0]):
			
			#loop for selecting unseen images only if it is in the unseen_labels indices
			if(int(images[i]['label']) in unseen_labels):
				unseen_dict.append(images[i])
			else:
				seen_dict.append(images[i])

		#clearing memory
		del images

	#seen and unseen dicts
	seen = []
	unseen = []

	for i in range(len(seen_dict)):
		#dynamic screen printing
		sys.stdout.write('\r Adding {}/{} images'.format(i,len(seen_dict)))
		sys.stdout.flush()

		#entry list containing image and the label
		entry = []
		im = seen_dict[i]['feature']   
		im_label = int(seen_dict[i]['label'])
		
		#appending image and label to entry
		entry.append(im)
		entry.append(im_label)

		#appending entry to seen list
		seen.append(entry)

	for i in range(len(unseen_dict)):
		#dynamic screen printing
		sys.stdout.write('\r Adding {}/{} images'.format(i,len(unseen_dict)))
		sys.stdout.flush()
		
		#entry list containing image and the label
		entry = []
		im = unseen_dict[i]['feature']
		im_label = int(unseen_dict[i]['label'])
		entry.append(im)
		entry.append(im_label)
		
		#appending entry to unseen list
		unseen.append(entry)

	#clearing memory
	del seen_dict
	del unseen_dict

	#returning according to the selected mode
	return np.array(seen), np.array(unseen)