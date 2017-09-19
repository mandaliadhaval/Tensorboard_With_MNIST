
# coding: utf-8

# In[ ]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import urllib
import os


# In[ ]:


filter_size1 = 5 #For 5x5 filter
num_filters1 = 16 #16 filters for 1st layer

filter_size2 = 5 # For 5x5 filter
num_filters2 = 36 #36 filters for 2nd layer

Input_neurons = 128


# In[ ]:


LOGDIR = '/LOG/2'
GIST_URL = 'https://gist.githubusercontent.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1/raw/dfb8ee95b010480d56a73f324aca480b3820c180'


# In[ ]:


os.chdir('C:/Users/dhavalma/AnacondaProjects/LOG/2')


# In[ ]:


get_ipython().magic('pwd')


# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


# In[ ]:


urllib.request.urlretrieve('https://gist.githubusercontent.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1/raw/bb6be73bd34211543ed16bc3a7f8e9646e5892ae','labels_1024.tsv')
urllib.request.urlretrieve('https://gist.githubusercontent.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1/raw/bb6be73bd34211543ed16bc3a7f8e9646e5892ae','sprite_1024.png')


# In[ ]:


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))


# In[ ]:



data.test.clas = np.argmax(data.test.labels, axis=1)


# In[ ]:


data.test.clas


# In[ ]:


data.test.labels[0:5, :]


# In[ ]:


# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10


# In[ ]:


def plot_images (images, clas_true, clas_pred=None):
    assert len(images)== len(clas_true)== 9
    
    #Create a 4x3 grid for images
    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, ax in enumerate(axes.flat):
        # Plot images
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        
        #Show both true and predicted classes
        if clas_pred is None:
            xlabel = "True : {0}".format(clas_true[i])
        else:
            xlabel = "True : {0}, Pred: {1}".format(clas_true[i], clas_pred[i])
                
        ax.set_xlabel(xlabel)
            
        #Remove Ticks from the Plot
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()


# In[ ]:


# Get images from test set
images = data.test.images[0:9]

#Get respective true class of the images above
clas_true = data.test.clas[0:9]

#Plot the images and labels
plot_images(images=images, clas_true=clas_true)


# In[ ]:


def var_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def var_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# In[ ]:


def conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True, name="Conv"):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = var_weights (shape = shape)
    bias = var_bias(length = num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += bias
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", bias)
    tf.summary.histogram("activations", layer)
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
        layer = tf.nn.relu(layer)
        return layer, weights


# In[ ]:


def flatten_layer(layer, name="flat_layer"):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


# In[ ]:


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True,
                 name="fc_layer"): # Use Rectified Linear Unit (ReLU)
    weights = var_weights(shape=[num_inputs, num_outputs])
    biases = var_bias(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)
    tf.summary.histogram("activations", layer)
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# In[ ]:


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


# In[ ]:


x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
tf.summary.image('input', x_image, 3)


# In[ ]:


y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


# In[ ]:


y_true_clas = tf.argmax(y_true, dimension=1)


# In[ ]:


layer_conv1, weights_conv1 =     conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True,
                   name="conv1")


# In[ ]:


layer_conv1


# In[ ]:


layer_conv2, weights_conv2 =     conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True,
                   name="conv2")


# In[ ]:


layer_conv2


# In[ ]:


layer_flat, num_features = flatten_layer(layer_conv2)


# In[ ]:


layer_flat


# In[ ]:


num_features


# In[ ]:


layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=Input_neurons,
                         use_relu=True)
embedding_input = layer_fc1
embedding_size = 128


# In[ ]:


layer_fc1


# In[ ]:


layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=Input_neurons,
                         num_outputs=num_classes,
                         use_relu=False)


# In[ ]:


layer_fc2


# In[ ]:


y_pred = tf.nn.softmax(layer_fc2)


# In[ ]:


y_pred_clas = tf.argmax(y_pred, dimension=1)


# In[ ]:


with tf.name_scope("cost"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true, name="Cross_Entropy")
    cost = tf.reduce_mean(cross_entropy, name="cost")
    tf.summary.scalar("cost", cost)


# In[ ]:


optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# In[ ]:


with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_clas, y_true_clas)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    


# In[ ]:


#summ = tf.summary.merge([tf.summary.scalar("cost", cost)])


# In[ ]:


embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
assignment = embedding.assign(embedding_input)
saver = tf.train.Saver()


# In[ ]:


sess = tf.Session()


# In[ ]:


sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(LOGDIR)# + hparam)
writer.add_graph(sess.graph)


# In[ ]:


config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding_config = config.embeddings.add()
embedding_config.tensor_name = embedding.name
embedding_config.sprite.image_path = LOGDIR + 'sprite_1024.png'
embedding_config.metadata_path = LOGDIR + 'labels_1024.tsv'
# Specify the width and height of a single thumbnail.
embedding_config.sprite.single_image_dim.extend([28, 28])
tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)


# In[ ]:


train_batch_size = 64


# In[ ]:


total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        sess.run(optimizer, feed_dict=feed_dict_train)
        
        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            #acc = sess.run(accuracy, feed_dict=feed_dict_train)
            acc = sess.run([accuracy, summ], feed_dict=feed_dict_train)
            writer.add_summary(acc, i)
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# In[ ]:


def plot_example_errors(clas_pred, correct):
    # This function is called from print_test_accuracy() below.

    # clas_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    clas_pred = clas_pred[incorrect]

    # Get the true classes for those images.
    clas_true = data.test.clas[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                clas_true=clas_true[0:9],
                clas_pred=clas_pred[0:9])


# In[ ]:


def plot_examples(clas_pred, correct):
    # This function is called from print_test_accuracy() below.

    # clas_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    correct = (correct == True)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images_correct = data.test.images[correct]
    
    # Get the predicted classes for those images.
    clas_pred = clas_pred[correct]

    # Get the true classes for those images.
    clas_true = data.test.clas[correct]
    
    # Plot the first 9 images.
    plot_images(images=images_correct[0:9],
                clas_true=clas_true[0:9],
                clas_pred=clas_pred[0:9])


# In[ ]:


def plot_confusion_matrix(clas_pred):
    # This is called from print_test_accuracy() below.

    # clas_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    clas_true = data.test.clas
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=clas_true,
                          y_pred=clas_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[ ]:


# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_examples=False, show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    clas_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        clas_pred[i:j] = sess.run(y_pred_clas, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    clas_true = data.test.clas

    # Create a boolean array whether each image is correctly classified.
    correct = (clas_true == clas_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_examples:
        print("Examples:")
        plot_examples(clas_pred=clas_pred, correct=correct)
        
    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(clas_pred=clas_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(clas_pred=clas_pred)


# In[ ]:


print_test_accuracy()


# In[ ]:


tf.reset_default_graph()


# In[ ]:


optimize(num_iterations=1)


# In[ ]:


print_test_accuracy()


# In[ ]:


optimize(num_iterations=99)


# In[ ]:


print_test_accuracy(show_example_errors=False)


# In[ ]:


print_test_accuracy(show_example_errors=True)


# In[ ]:


print_test_accuracy(show_examples=False)


# In[ ]:


print_test_accuracy(show_examples=True, show_example_errors=True)


# In[ ]:


optimize(num_iterations=900)


# In[ ]:


print_test_accuracy(show_examples=True, show_example_errors=True)


# In[ ]:


optimize(num_iterations=9000)


# In[ ]:


print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


# In[ ]:


def plot_conv_weights(weights, input_channel=0):
# Initialize weight
    w = sess.run(weights)

    
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    
    num_grids = math.ceil(math.sqrt(num_filters))
    
    
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    
    plt.show()


# In[ ]:


def plot_conv_layer(layer, image):
  
    feed_dict = {x: [image]}

   
    values = sess.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    num_grids = math.ceil(math.sqrt(num_filters))
    
    
    fig, axes = plt.subplots(num_grids, num_grids)

    
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()


# In[ ]:


plot_conv_weights(weights=weights_conv1)


# In[ ]:


def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()


# In[ ]:


image1 = data.test.images[1]
plot_image(image1)


# In[ ]:


image2 = data.test.images[13]
plot_image(image2)


# In[ ]:


plot_conv_weights(weights=weights_conv1)


# In[ ]:


plot_conv_layer(layer=layer_conv1, image=image1)


# In[ ]:


plot_conv_layer(layer=layer_conv1, image=image2)


# In[ ]:


plot_conv_weights(weights=weights_conv2, input_channel=0)


# In[ ]:


plot_conv_weights(weights=weights_conv2, input_channel=1)


# In[ ]:


plot_conv_layer(layer=layer_conv2, image=image1)


# In[ ]:


plot_conv_layer(layer=layer_conv2, image=image2)


# In[ ]:


#sess.close()


# In[ ]:


optimize(num_iterations=10000)


# In[ ]:


print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


# In[ ]:




