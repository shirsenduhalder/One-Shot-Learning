import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 

class Model:

    def __init__(self, session, data_dim, label_dim, latent_dim, output_file, batch_size=50, weights=None, biases=None):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.session = session
        self.batch_size = batch_size
        self.output_file = output_file

        self.define_placeholders()
        self.define_weights(weights, biases)
        self.define_ops()

    def define_placeholders(self):
        self.X11 = tf.placeholder(tf.float32, shape=[None,self.data_dim], name='Seen_vis')
        self.X12 = tf.placeholder(tf.float32, shape=[None,self.data_dim], name='Seen_self_vis')
        self.X13 = tf.placeholder(tf.float32, shape=[None, self.data_dim], name='Seen_cross_class')

        self.Y1 = tf.placeholder(tf.float32, shape=[None, self.label_dim], name='Seen_label')

    def define_weights(self, weights, biases):
        if weights and biases:
            weights = self.extend_weights(weights)
            biases = self.extend_weights(biases, type='bias')

            self.W11 = tf.Variable(weights["W11"].astype(np.float32), name = "W11")
            self.W31 = tf.Variable(weights["W31"].astype(np.float32), name = "W31")

            self.bias11 = tf.Variable(biases["bias11"].astype(np.float32), name='bias11')
            self.bias12 = tf.Variable(biases["bias12"].astype(np.float32), name='bias12')
            self.bias31 = tf.Variable(biases["bias13"].astype(np.float32), name='bias31')

        else:
            self.W11 = tf.Variable(tf.random_normal([self.data_dim,self.latent_dim]), name='W11')
            self.W31 = tf.Variable(tf.random_normal([self.latent_dim, self.label_dim]), name='W31')

            self.bias11 = tf.Variable(tf.random_normal([self.latent_dim]), name='bias11')
            self.bias12 = tf.Variable(tf.random_normal([self.data_dim]), name='bias12')
            self.bias31 = tf.Variable(tf.random_normal([self.label_dim]), name='bias31')


    def extend_weights(self, weights, type='weights'):
        if type=='weights':
            extended_weights = np.random.normal(size=(self.latent_dim, self.label_dim))
            extended_weights[:,:-1] = weights["W31"]

            weights["W31"] = extended_weights
            return weights
        else:
            bias_extended_weights = np.random.normal(size=(self.label_dim))
            bias_extended_weights[:-1] = weights["bias13"]

            weights["bias13"] = bias_extended_weights
            return weights

    def define_ops(self):
        latent_seen = tf.nn.relu(tf.add(tf.matmul(self.X11,self.W11),self.bias11), name = 'latent_seen')

        rec_seen = tf.nn.relu(tf.add(tf.matmul(latent_seen,tf.transpose(self.W11)),self.bias12), name = 'auto_seen')
        rec_cross_class = tf.nn.relu(tf.add(tf.matmul(latent_seen, tf.transpose(self.W11)), self.bias12), name = 'source_cross_seen')

        self.logit  = tf.add(tf.matmul(latent_seen,self.W31),self.bias31, name = 'vis_class')
        class_loss = tf.nn.softmax_cross_entropy_with_logits(logits = logit, labels = self.Y1)

        classifier_loss_t = 1000*tf.reduce_mean(class_loss) + tf.norm(self.W31) + tf.norm(self.bias31)

        source_loss = tf.norm(tf.subtract(rec_seen, self.X12))  - tf.norm(tf.subtract(rec_cross_class, self.X13)) + tf.norm(self.W11) + tf.norm(self.bias11) + tf.norm(self.bias12)

        self.total_loss = classifier_loss_t + source_loss

        self.acc = tf.equal(tf.argmax(tf.nn.softmax(logit), 1), tf.argmax(self.Y1, 1))
        self.acc = tf.reduce_mean(tf.cast(self.acc, tf.float32))

        self.optimizer=tf.train.AdamOptimizer(0.01).minimize(self.total_loss)
        self.init=tf.global_variables_initializer()

    def train_graph(self, triplet_input, triplet_self, triplet_cross, triplet_labels, epochs = 1):
        total_batch = int(triplet_input.shape[0]/self.batch_size)+1
        batches_seen1, batches_seen2, batches_seen3, batches_seen_label = self.get_batches_seen(triplet_input, triplet_self, triplet_cross, triplet_labels, total_batch, self.batch_size, 'train') 

        # print(batches_seen1.shape)
        # print(batches_seen2.shape)
        # print(batches_seen_label.shape)

        self.session.run(self.init)
        weights = None
        biases = None
        final_acc = 0
        final_cost = 0
        for epoch in range(epochs):

            avg_cost = 0
            avg_acc = 0
            
            for i in range(total_batch):
                batch_x11 = np.array(batches_seen1[i]).astype(np.float32)
                batch_x12 = np.array(batches_seen2[i]).astype(np.float32)
                batch_x13 = np.array(batches_seen3[i]).astype(np.float32)
                batch_x1_label = np.array(batches_seen_label[i]).astype(np.float32)
                
                _ , c, acc_new = self.session.run([self.optimizer, self.total_loss, self.acc], feed_dict = {self.X11: batch_x11, self.X12: batch_x12, self.X13: batch_x13, self.Y1: batch_x1_label})

                avg_cost += c /total_batch
                avg_acc += acc_new /total_batch

                #print(avg_cost, avg_acc)

            final_cost = avg_cost
            final_acc = avg_acc

            print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost), "accuracy =", "{:.5f}".format(avg_acc), file=self.output_file)

            weights = self.session.run({"W11": self.W11, "W31":self.W31})
            biases = self.session.run({"bias11":self.bias11, "bias12":self.bias12, "bias13":self.bias31})

        print ("Training of seen classes complete!", file=self.output_file)
        return weights, biases, final_acc, final_cost

    def test_model_accuracy(self, testing_input, testing_labels):
        acc_new = self.session.run([self.acc], feed_dict = {self.X11: testing_input, self.Y1: testing_labels})
        return acc_new
        
    def debug_values(self, testing_input, testing_labels):
        output = self.session.run([self.logit], feed_dict = {self.X11 : testing_input})
        return output

    def close_session(self):
        self.session.close()

    def get_batches_seen(self, triplet_input, triplet_self, triplet_cross, triplet_labels, total_batch, batch_size, name):

        batches1 = []
        batches2 = []
        batches3 = []
        label = []

        if total_batch == 0:
            return [triplet_input[:,:], triplet_self[:,:], triplet_cross[:,:], triplet_labels]
        
        for i in range(total_batch):
            temp_batch = [x for x in triplet_input[i*batch_size:(i+1)*batch_size, :]]
            batches1.append(temp_batch)

            temp_batch = [x for x in triplet_self[i*batch_size:(i+1)*batch_size, :]]
            batches2.append(temp_batch)

            temp_batch = [x for x in triplet_cross[i*batch_size:(i+1)*batch_size, :]]
            batches3.append(temp_batch)

            label.append(triplet_labels[i*batch_size:(i+1)*batch_size, :])
        
        return np.array(list(batches1)), np.array(list(batches2)), np.array(list(batches3)), np.array(list(label))