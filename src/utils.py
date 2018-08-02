import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
import random

def fix_off_by_one(labels):
    labels = np.array(labels)
    print(labels.shape)
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

def catas_forg(last_data, this_data_size, ratio=1.0):
    if last_data == []:
        return []
    print(ratio*this_data_size)
    new_data_indices = random.sample(range(0,last_data[0].shape[0]), int(this_data_size*ratio))
    print(new_data_indices)
    iin = []
    out_self = []
    out_recons = []
    label = []

    for x in new_data_indices:
        iin.append(last_data[0][x])
        out_self.append(last_data[1][x])
        out_recons.append(last_data[2][x])
        label.append(last_data[3][x])

    return np.array(iin), np.array(out_self), np.array(out_recons), np.array(label)


def merge_data(last_data, new_data, randomize=False):
    if last_data == []:
        return new_data
    if randomize:
        iin = []
        out_self = []
        out_recons = []
        label = []
        rl = [1]*new_data[0].shape[0] + [0]*last_data[0].shape[0]
        random.shuffle(rl)
        i = 0
        j = 0
        for t in rl:
            if t == 0:
                iin.append(last_data[0][i])
                out_self.append(last_data[1][i])
                out_recons.append(last_data[2][i])
                label.append(last_data[3][i])
                i+=1
            else:
                iin.append(new_data[0][j])
                out_self.append(new_data[1][j])
                out_recons.append(new_data[2][j])
                label.append(new_data[3][j])
                j+=1
        return np.array(iin), np.array(out_self), np.array(out_recons), np.array(label)
    else:    
        merged = []
        merged.append(np.vstack((last_data[0], new_data[0])))
        merged.append(np.vstack((last_data[1], new_data[1])))
        merged.append(np.vstack((last_data[2], new_data[2])))
        merged.append(np.concatenate((last_data[3], new_data[3])))
        return merged

