import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import rnn
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl as rnn_cell
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
import attention_encoder
import Generate_stock_data as GD
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable Tensorflow debugging message

def RNN(encoder_input, decoder_input, weights, biases, encoder_attention_states):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Prepare data for encoder
    # Permuting batch_size and n_steps
    encoder_input = tf.transpose(encoder_input, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    encoder_input = tf.reshape(encoder_input, [-1, n_input_encoder])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    encoder_input = tf.split(encoder_input, n_steps_encoder, 0)

    # Prepare data for decoder
    # Permuting batch_size and n_steps
    decoder_input = tf.transpose(decoder_input, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    decoder_input = tf.reshape(decoder_input, [-1, n_input_decoder])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    decoder_input = tf.split(decoder_input, n_steps_decoder,0 )

    # Encoder.
    with tf.variable_scope('encoder') as scope:
        encoder_cell = rnn_cell.BasicLSTMCell(n_hidden_encoder, forget_bias=1.0)
        encoder_outputs, encoder_state, attn_weights = attention_encoder.attention_encoder(encoder_input,
                                         encoder_attention_states, encoder_cell)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [tf.reshape(e, [-1, 1, encoder_cell.output_size]) for e in encoder_outputs]
    attention_states = tf.concat(top_states,1)

    with tf.variable_scope('decoder') as scope:
        decoder_cell = rnn_cell.BasicLSTMCell(n_hidden_decoder, forget_bias=1.0)
        outputs, states = seq2seq.attention_decoder(decoder_input, encoder_state,
                                            attention_states, decoder_cell)

    return tf.matmul(outputs[-1], weights['out1']) + biases['out1'], attn_weights

all_pred_sign = []
all_test_val = []
num_accu = 0
period =  1851 #21

for i in range(0, period):
    tf.reset_default_graph()

    print i
    # Parameters
    learning_rate = 0.001
    training_iters = 1000 #50000
    batch_size = 48 # 64 #128 #48
    display_step = 100
    model_path = "./stock_dual/"

    # Network Parameters
    # encoder parameter
    num_feature =  47 # number of index #47 #231 #81
    n_input_encoder =  47 # n_feature of encoder input #47 #231 #81
    n_steps_encoder = 2 # time steps #10
    n_hidden_encoder = 64 # size of hidden units #128

    # decoder parameter
    n_input_decoder = 1
    n_steps_decoder = 1 # 9
    n_hidden_decoder = 64 #128
    n_classes = 1 # size of the decoder output

    # tf Graph input
    encoder_input = tf.placeholder("float", [None, n_steps_encoder, n_input_encoder])
    decoder_input = tf.placeholder("float", [None, n_steps_decoder, n_input_decoder])
    decoder_gt = tf.placeholder("float", [None, n_classes])
    encoder_attention_states = tf.placeholder("float", [None, n_input_encoder, n_steps_encoder])

    # Define weights
    weights = {'out1': tf.Variable(tf.random_normal([n_hidden_decoder, n_classes]))}
    biases = {'out1': tf.Variable(tf.random_normal([n_classes]))}


    pred, attn_weights = RNN(encoder_input, decoder_input, weights, biases, encoder_attention_states)
    # Define loss and optimizer
    cost = tf.reduce_sum(tf.pow(tf.subtract(pred, decoder_gt), 2))
    loss = tf.pow(tf.subtract(pred, decoder_gt), 2)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    # save the model
    saver = tf.train.Saver()
    loss_value = []
    step_value = []
    loss_test=[]
    loss_val = []

    # Launch the graph


    with tf.Session() as sess:
        sess.run(init)
        step = 1
        count = 1

        # read the input data
        Data = GD.Input_data(batch_size, n_steps_encoder, n_steps_decoder, n_hidden_encoder, i)
        # Keep training until reach max iterations
        while step  < training_iters:
            
            # the shape of batch_x is (batch_size, n_steps, n_input)
            batch_x, batch_y, prev_y, encoder_states = Data.next_batch()
            feed_dict = {encoder_input: batch_x, decoder_gt: batch_y, decoder_input: prev_y,
                        encoder_attention_states:encoder_states}
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict)
            # display the result
            if step % display_step == 0:
                # Calculate batch loss
                loss = sess.run(cost, feed_dict)/batch_size
                # print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss)

                #store the value
                loss_value.append(loss)
                step_value.append(step)
                # Val
                val_x, val_y, val_prev_y, encoder_states_val = Data.validation()
                feed_dict = {encoder_input: val_x, decoder_gt: val_y, decoder_input: val_prev_y,
                            encoder_attention_states:encoder_states_val}
                loss_val1 = sess.run(cost, feed_dict)/len(val_y)
                loss_val.append(loss_val1)
                # print "validation Accuracy:", loss_val1

                # testing
                test_x, test_y, test_prev_y, encoder_states_test= Data.testing()
                feed_dict = {encoder_input: test_x, decoder_gt: test_y, decoder_input: test_prev_y,
                            encoder_attention_states:encoder_states_test}
                pred_y=sess.run(pred, feed_dict)
                loss_test1 = sess.run(cost, feed_dict)/len(test_y)
                loss_test.append(loss_test1)
                # print "Testing Accuracy:", loss_test1

                #save the parameters
                if loss_val1<=min(loss_val):
                    save_path = saver.save(sess, model_path  + 'dual_stage_' + str(step) + '.ckpt')
            
            step += 1
            count += 1

            # reduce the learning rate
            if count > 10000:
                learning_rate *= 0.1
                count = 0
                save_path = saver.save(sess, model_path  + 'dual_stage_' + str(step) + '.ckpt')


        mean, stdev = Data.returnMean()
        testing_result = test_y*stdev[num_feature] + mean[num_feature]
        pred_result = pred_y*stdev[num_feature] + mean[num_feature]

       
        

        testing_sign = []
        pred_sign = []
        ind = len(testing_result)-1

        if testing_result[ind] > testing_result[ind-1]:
            testing_sign.append(1)
        elif testing_result[ind] < testing_result[ind-1]:
            testing_sign.append(-1)
        else:
            testing_sign.append(0)

        if pred_result[ind] > pred_result[ind-1]:
                pred_sign.append(1)
                all_pred_sign.append(1)
        elif pred_result[ind] < pred_result[ind-1]:
            pred_sign.append(-1)
            all_pred_sign.append(-1)
        else:
            pred_sign.append(0)
            all_pred_sign.append(0)


        for x in range(0, len(pred_sign)):
            if testing_sign[x] == pred_sign[x]:
                num_accu += 1

        print "testing data:"
        print testing_result

        print "prediction data:"
        print pred_result

        all_test_val.append(testing_result)

        print testing_sign
        print pred_sign

        print all_pred_sign



accuracy = float(num_accu)/float(period+1)
print "Accuracy for %d day(s): %f" %(period+1, accuracy)

df = pd.DataFrame(all_pred_sign, columns=["pred_sign"])
df.to_csv('pred_sign.csv', index=False)

    







