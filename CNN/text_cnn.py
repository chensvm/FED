# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        '''
        sequence_length – The length of our sentences. Remember that we padded all our sentences to have the same length (59 for our data set).
        num_classes – Number of classes in the output layer, two in our case (positive and negative).
        vocab_size – The size of our vocabulary. This is needed to define the size of our embedding layer, which will have shape [vocabulary_size, embedding_size].
        embedding_size – The dimensionality of our embeddings.
        filter_sizes – The number of words we want our convolutional filters to cover. We will have num_filters for each size specified here. For example, [3, 4, 5] means that we will have filters that slide over 3, 4 and 5 words respectively, for a total of 3 * num_filters filters.
        num_filters – The number of filters per filter size (see above)
        '''

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        '''
        the shape of the input tensor. 
        None means that the length of that dimension could be anything. 
        In our case, the first dimension is the batch size, 
        and using None allows the network to handle arbitrarily sized batches.
        '''


        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #  we enable dropout only during training


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # build up word embedding
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # forces an operation to be executed on the CPU
            # tf.name_scope creates a new Name Scope with the name “embedding”. 
            # The scope adds all operations into a top-level node called “embedding” 
            # so that you get a nice hierarchy when visualizing your network in TensorBoard.

            # W is our embedding matrix that we learn during training.
            # using a random uniform distribution
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # embedding_lookup creates the actual embedding operation
            #  The result of the embedding operation is a 
            # 3-dimensional tensor of shape [None, sequence_length, embedding_size].
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            '''
             conv2d operation expects a 4-dimensional tensor 
             batch, width, height and channel
             但是在這裡我們的channel手動設為[None, sequence_length, embedding_size, 1]
            '''

        # Create a convolution + maxpool layer for each filter size
        # 使用max pooling做這件事，所以不管filter size多大，output出來的結果會一樣
        # 然後merge the results into one big feature vector

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #  W is our filter matrix
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    # slide the filter over our sentence without padding the edges
                    # narrow convolution
                    # gives us an output of shape [1, sequence_length - filter_size + 1, 1, 1]

                    name="conv")

                # 進行max pooling leaves us with a tensor of shape [batch_size, 1, 1, num_filters]
                # 就是一個feature vector


                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # the result of applying the nonlinearity to the convolution output

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Using -1 in tf.reshape tells TensorFlow to flatten the dimension when possible
        

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
