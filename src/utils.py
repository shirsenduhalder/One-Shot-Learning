import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 

def fix_off_by_one(labels):
    labels = np.array(labels)
    labels -= 1
    return labels

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

def one_hot(labels, one_hot_size):
    a = np.array(labels)
    b = np.zeros((a.size, one_hot_size))
    b[np.arange(a.size),a] = 1
    return b

def club_data(data):
    iin = []
    out_self = []
    out_recons = []
    label = []

    for x in data:
        iin.extend(x[0])
        out_self.extend(x[1])
        out_recons.extend(x[2])
        label.extend(x[3])

    return np.array(iin), np.array(out_self), np.array(out_recons), fix_off_by_one(label)

def club_test_data(data):
    iin = []
    label = []

    for x in data:
        #print(x.shape)
        #print(x[:,0].shape)
        #print(x[:,1].shape)
        iin.extend(x[:,0])
        label.extend(x[:,1])

    return np.array(iin), fix_off_by_one(label)

def take_random(data):
    pass