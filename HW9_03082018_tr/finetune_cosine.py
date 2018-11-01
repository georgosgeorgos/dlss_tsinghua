import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.data import Iterator

# Path to the textfiles for the trainings and validation set
flag = "50"
train_file = '../data/caltech_train_' + flag + 'images.txt'
val_file = '../data/caltech_test.txt'

# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = int(flag)

# Network params
dropout_rate = 0.5
num_classes = 50
train_layers = ['fc7', 'fc8']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "../record/tensorboard"


# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=False)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# ---------- Now you should complete the code below ------------

# Task 2: List of trainable variables of the layers we want to train
# Task 2: Search for usage of tf.trainable_variables()
# Task 2: You should get all of the trainable_variables, and find those variables whose name matches train_layers(='fc8'), 
# then store it to the var_list.
# Task 2: Substitute some codes to ???
variables = tf.trainable_variables()
var_list = []
for i in variables:
    if (i.name.split("/")[0] in ["fc7", "fc8"]):
        var_list.append(i)
        
# Train op
with tf.name_scope("train"):
    # Task 2: Get gradients of all trainable variables
    # Task 2: Search for usage of tf.gradient() and use zip() function to build one-to-one mapping for the gradient
    # and the variable.
    gradients = []
    for i in var_list:
        gradients.append(tf.gradients(loss, [i])[0])
        
    gradients = zip(gradients, var_list)
    print(gradients)

    # Task 2: Create optimizer and apply gradient descent to the trainable variables
    # Task 2: Search for usage of tf.train.GradientDescentOptimizer() and apply_gradients()
    # Task 3: An easy way to solve the different learning rate method is to design two optimizers and two train_ops
    # and apply two different learning rates to these two optimizers. Finally using tf.group() to group two train_ops
    # Task 3: Pay attention to that different train_op contains gradients corresponding to different variables.
    optimizer7 = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op7 = optimizer7.apply_gradients(gradients[0:1])
    
    optimizer8 = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op8 = optimizer8.apply_gradients(gradients[2:3])
    
    train_op = tf.group([train_op7, train_op8])
    
    fc7 = model.fc7
    
# ---------- End of your code ---------------

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    list_val = []
    list_train50 = []
    
    sess.run(training_init_op)
    for step in range(train_batches_per_epoch):

        # get next batch of data
        img_batch, label_batch = sess.run(next_batch)

        # And run the training op
        feat = sess.run(fc7, feed_dict={x: img_batch,keep_prob: 1})

        list_train50.append(np.mean(feat, axis=0))
        
    array_train50 = np.array(list_train50)
    np.save("array_train"+flag, array_train50)

    # Validate the model on the entire validation set
    print("{} Start validation".format(datetime.now()))
    sess.run(validation_init_op)
    for _ in range(val_batches_per_epoch):

        img_batch, label_batch = sess.run(next_batch)
        feat = sess.run(fc7, feed_dict={x: img_batch, keep_prob: 1.})
        list_val.extend(feat)
        
    array_val = np.array(list_val)
    np.save("array_val", array_val)
