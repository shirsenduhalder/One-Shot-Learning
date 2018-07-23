import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
import time
import os 
from datetime import timedelta
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

from utils import *
from im_model import *

## Define data paths
train_seen_path = "train_seen.npy"
train_unseen_path = "train_unseen.npy"
test_seen_path = "test_seen.npy"
test_unseen_path = "test_unseen.npy"


#load data
train_seen = np.load(train_seen_path)
train_unseen = np.load(train_unseen_path)
test_seen = np.load(test_seen_path)
test_unseen = np.load(test_unseen_path)

## Define constants
N_INITIAL_CLASSES = 15
SEEN_CLASSES = train_seen.shape[0]
UNSEEN_CLASSES = train_unseen.shape[0]
TOTAL_CLASSES = SEEN_CLASSES + UNSEEN_CLASSES
BATCH_SIZE = 32
INPUT_SIZE = 2048
LATENT_SIZE = 450
CATASTROPHIC_FORGETTING_SAMPLING_RATIO = 1.0
LEARNING_RATE = 0.01
CLASSIFIER_LOSS_WEIGHT = 1000
LABEL_SIZE = N_INITIAL_CLASSES


last_model_weights = ()
cumulative_last_data = []

i = N_INITIAL_CLASSES

for i in range(N_INITIAL_CLASSES, TOTAL_CLASSES+1):
    #output file
    f = open("accuracy_outputs.txt", "a+")

    print("### STARTING NEW CYCLE %s ###"%str(i), file=f) 
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    if i == N_INITIAL_CLASSES:
        class_type = "initial"
        step = i
        train_data = club_data(train_seen[0:N_INITIAL_CLASSES])
        test_data = club_test_data(test_seen[0:N_INITIAL_CLASSES])
        model = Model(sess, INPUT_SIZE, LABEL_SIZE, LATENT_SIZE, f, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, classifier_loss_weight= CLASSIFIER_LOSS_WEIGHT)

    elif i <= SEEN_CLASSES:
        LABEL_SIZE += 1
        class_type = "seen"
        step = i - N_INITIAL_CLASSES
        train_data = club_data(train_seen[i-1:i])
        test_data = club_test_data(test_seen[0:i])
        model = Model(sess, INPUT_SIZE, LABEL_SIZE, LATENT_SIZE, f, batch_size=BATCH_SIZE, weights=last_model_weights[0], biases=last_model_weights[1], learning_rate=LEARNING_RATE, classifier_loss_weight= CLASSIFIER_LOSS_WEIGHT)

    else:
        LABEL_SIZE += 1
        class_type = "unseen"
        step = i - SEEN_CLASSES
        train_data = club_data(train_unseen[i-1-SEEN_CLASSES:i-SEEN_CLASSES])
        test_data = club_test_data(test_unseen[0:i-SEEN_CLASSES])
        model = Model(sess, INPUT_SIZE, LABEL_SIZE, LATENT_SIZE, f, batch_size=BATCH_SIZE, weights=last_model_weights[0], biases=last_model_weights[1], learning_rate=LEARNING_RATE, classifier_loss_weight= CLASSIFIER_LOSS_WEIGHT)
    
    catas_forget_data = catas_forg(cumulative_last_data, train_data[0].shape[0], ratio=CATASTROPHIC_FORGETTING_SAMPLING_RATIO)    
    final_train_data_for_this_class = merge_data(catas_forget_data, train_data)
    cumulative_last_data = merge_data(cumulative_last_data, train_data)

    print('LABEL_SIZE: {}'.format(LABEL_SIZE), file=f)   
    train_labels = one_hot((train_data[3]), LABEL_SIZE)
    test_labels = one_hot(test_data[1], LABEL_SIZE)

    weights, biases, acc, cost = model.train_graph(train_data[0], train_data[1], train_data[2], train_labels, epochs=5)
    test_acc = model.test_model_accuracy(test_data[0], test_labels)

    last_model_weights = (weights, biases)
    print("------ ------- Type : %s ; Accuracy at step %s - Train %s , Test %s with label size %s" % (class_type, str(step), str(acc), str(test_acc), str(LABEL_SIZE)), file=f)
    model.close_session()

    print("### END OF CYCLE %s ###"%str(i), file=f)
    f.close()