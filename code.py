# importing dependencies.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# reading the data from the csv file.
raw_data = []
raw_labels = []

file = open('dataset.csv')
read_file = file.readline().rstrip("\n")

while read_file:
	values = read_file.split(",")
	values = [float(i) for i in values]
	raw_data.append(values[0:-1])
	
	label = int(values[-1])
	if label == 0:
		raw_labels.append([0])
	else:
		raw_labels.append([1])
	
	read_file = file.readline().rstrip("\n")
    
file.close()

# splitting the data into training and testing data.
# training and testing data.

train_data = raw_data[0:500]
test_data = raw_data[501:]
train_label = raw_labels[0:500]
test_label = raw_labels[501:]
loss_trace = []

# specifications of the nn.
n_nodes_hl1 = 4
n_classes = 1
learning_rate = 1e-3

# declaring placeholders.
x = tf.placeholder('float', [None, 4])
y = tf.placeholder('float')

# defining the computation graph.

def neural_network_model(data):
    
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([4, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.sigmoid(l1)
    
    output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])
    
    return output


# training the neural network.

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = prediction))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    noofepochs = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            loss = 0 
            for _ in range(noofepochs):
                t ,c = sess.run([optimizer, cost], feed_dict = {x:train_data, y:train_label})
                loss = loss + c
                loss_trace.append(c)
            print('step:',epoch,'cost:',loss)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy',accuracy.eval({x: test_data, y: test_label}))
        

# running the program.

train_neural_network(x)


# visualising the loss curve with no of epochs.
plt.plot(loss_trace)
plt.title('Cross Entropy loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
