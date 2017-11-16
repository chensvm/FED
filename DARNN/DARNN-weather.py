# coding: utf-8

# check whether it is updated

import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.models.rnn import *
from tensorflow.contrib import rnn
import os




data = 'data/weather_30000.csv'
df = pd.read_csv(data)
df.head()
# In[3]:


#for col in list(df.columns)
cols = list(df.columns)
print cols
#print len(df)
#print len(df.loc[~(df['VisibilityM']==-9999)])


# In[4]:


def preprocess(df):   
    cols = ['TempM', 'DewPointM', 'Humidity', 'WindSpeedM', 'VisibilityM', 'PressureA']
    df = df[cols]
    for col in cols:
        df = df.loc[~(df[col]==-9999)]
        df = df[~(df.isnull())]
    df = df.reset_index(drop = True)
    return df


# In[22]:


df1 = preprocess(df)
df1.describe()


# In[ ]:


scaler = MinMaxScaler((0,1))
def get_batch(df,batch_size = 128, T = 16, input_dim = 6, step = 0, train = True):
    
    t = step*batch_size
    X_batch = np.empty(shape = [batch_size,T,input_dim])
    y_batch = np.empty(shape = [batch_size,T])
    labels = np.empty(shape = [batch_size])
    
    for i in range(batch_size):
        X_batch[i,:] = scaler.fit_transform(df.iloc[t:t+T].values)
        y_batch[i,:] = df["TempM"].iloc[t:t+T].values
        labels[i] = df["TempM"].iloc[t+T]
        t += 1     
    
    ## shuffle in train, not in test
    if train:
        index = range(batch_size)
        np.random.shuffle(index)
        X_batch = X_batch[index]
        y_batch = y_batch[index]
        labels = labels[index]

    return X_batch,y_batch,labels


# In[56]:


class ts_prediction(object):
    
    def __init__(self, input_dim, time_step, n_hidden, d_hidden, batch_size):

        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.d_hidden = d_hidden
        self.o_hidden = 32
        
        self.input_dim = input_dim
        self.time_step = time_step
        
        self.seq_len = tf.placeholder(tf.int32,[None])
        self.input_x = tf.placeholder(dtype = tf.float32, shape = [None, None, input_dim])
        self.input_y = tf.placeholder(dtype = tf.float32,shape = [None,self.time_step])
        self.label = tf.placeholder(dtype = tf.float32)
        
        self.encode_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        self.decode_cell = tf.contrib.rnn.LSTMCell(self.d_hidden, forget_bias=1.0, state_is_tuple=True)
        self.output_cell = tf.contrib.rnn.LSTMCell(self.o_hidden, forget_bias=1.0, state_is_tuple=True)
        
        self.loss = tf.constant(0.0)
            ## ===========  build the model =========== ##
            
            ## ==== encoder ===== ## 
        out = self.en_RNN(self.input_x) # out[0]: (b*T*2d)
        self.out = out
        out = tf.transpose(out,[0,2,1]) # (b,2d,T)
        with tf.name_scope('encoder') as scope:
            stddev = 1.0/(self.n_hidden*self.time_step)
            Ue = tf.Variable(dtype=tf.float32,
                             initial_value = tf.truncated_normal(shape = [self.time_step,self.time_step], 
                                                                 mean = 0.0, stddev = stddev),name = 'Ue')
        var = tf.tile(tf.expand_dims(Ue,0),[self.batch_size,1,1]) #(b,T,T)
        
        batch_mul = tf.matmul(var,self.input_x) # (b*T*T)*(b*T*d) = (b,T,d)
        self.out = batch_mul
        e_list = []

        for k in range(self.input_dim):
            series_k = tf.reshape(batch_mul[:,:,k],[self.batch_size,self.time_step,1]) #(b,T,1)
            e_k = self.attention(out,series_k, scope = 'encoder')
            e_list.append(e_k)
        e_list = tf.concat(e_list,axis = 1)
        soft_attention = tf.nn.softmax(e_list,dim = 1)
        input_attention = tf.multiply(self.input_x,tf.transpose(soft_attention,[0,2,1])) #(b,T,d)
        
        with tf.variable_scope('fw_lstm') as scope:
            tf.get_variable_scope().reuse_variables()
            h,_ = tf.nn.dynamic_rnn(self.encode_cell, input_attention, self.seq_len, dtype = tf.float32)
        # h: (b,T,d)
        
            # ===== decoder ===== ## 
        d, dec_out = self.de_RNN(h) ## d: (b,T,q); dec_out:(b,T,2q)
        self.out = d
        dec_out = tf.transpose(dec_out,[0,2,1])
        with tf.name_scope('decoder') as scope:
            stddev = 1.0/(self.d_hidden*self.time_step)
            Ud = tf.Variable(dtype=tf.float32,
                             initial_value = tf.truncated_normal(shape = [self.n_hidden,self.n_hidden], 
                                                                 mean = 0.0, stddev = stddev),name = 'Ud')
        de_var = tf.tile(tf.expand_dims(Ud,0),[self.batch_size,1,1]) # (b,d,d)
        batch_mul_de = tf.matmul(h,de_var) #(b, T, d)
        batch_mul_de = tf.transpose(batch_mul_de,[0,2,1])
        e_de_list = []
        for t in range(self.time_step):
            series_t = tf.reshape(batch_mul_de[:,:,t],[self.batch_size,self.n_hidden,1])
            e_t = self.de_attention(dec_out,series_t, scope = 'decoder')
            e_de_list.append(e_t)
        e_de_list = tf.concat(e_de_list,axis = 1) # b,T,T
        de_soft_attention = tf.nn.softmax(e_de_list,dim = 1)
        #self.out = de_soft_attention
        
            # ===== context c_t ===== ##
        c_list = []
        for t in range(self.time_step):
            Beta_t = tf.expand_dims(de_soft_attention[:,:,0],-1)
            weighted = tf.reduce_sum(tf.multiply(Beta_t,h),1)
            c_list.append(tf.expand_dims(weighted,1))
        c_t = tf.concat(c_list,axis = 1) ## (b,T,d)
        self.out = c_t
        c_t_hat = tf.concat([c_t,tf.expand_dims(self.input_y,-1)],axis = 2) # b,T,(d+1), where +1 for concatenation
        
            # ===== y_hat ===== ##
        with tf.variable_scope('temporal'):
            mean = 0.0
            stddev = 1.0/(self.n_hidden*self.time_step)
            W_hat = tf.get_variable(name = 'W_hat',shape = [self.n_hidden+1,1],dtype = tf.float32,
                                    initializer=tf.truncated_normal_initializer(mean,stddev)) 
        
        W_o = tf.tile(tf.expand_dims(W_hat,0),[self.batch_size,1,1])
        y_hat = tf.matmul(c_t_hat,W_o) ## b,T,1
        
            ## ==== final step ==== ##
        d_y_concat = tf.concat([d,y_hat],axis = 2) ## b,T,q+1
        with tf.variable_scope('out_lstm') as scope:
            d_final,_ = tf.nn.dynamic_rnn(self.output_cell, d_y_concat, self.seq_len, dtype = tf.float32) # b,T,o_hidden
            
            ## ==== output y_T ==== ##
        ## only concat the last state d_T and c_T
        d_c_concat = tf.concat([d_final[:,-1,:],c_t[:,-1,:]],axis = 1) #b,o_hidden+q
        d_c_concat = tf.expand_dims(d_c_concat,-1) # b,d+q,1
        
        with tf.variable_scope('predict'):
            mean = 0.0
            stddev = 1.0/(self.n_hidden*self.time_step)
            Wy = tf.get_variable(name = 'Wy',shape = [self.o_hidden,self.o_hidden+self.n_hidden],dtype = tf.float32,
                                 initializer=tf.truncated_normal_initializer(mean,stddev)) 
            Vy = tf.get_variable(name = 'Vy',shape = [self.o_hidden],dtype = tf.float32,
                                 initializer=tf.truncated_normal_initializer(mean,stddev)) 
            bw = tf.get_variable(name = 'bw',shape = [self.o_hidden],dtype = tf.float32,
                                initializer = tf.constant_initializer(0.1))
        W_y = tf.tile(tf.expand_dims(Wy,0),[self.batch_size,1,1]) # b,q,q+d
        b_w = tf.expand_dims(tf.tile(tf.expand_dims(bw,0),[self.batch_size,1]),-1) #b,q -> b,q,1
        V_y = tf.tile(tf.expand_dims(Vy,0),[self.batch_size,1]) #b,q
        V_y = tf.expand_dims(V_y,1) #b,1,q
        self.y_predict = tf.squeeze(tf.matmul(V_y,tf.matmul(W_y,d_c_concat)+b_w)) #(b,1,q) * (b,q,1) -> squeeze -> (b,)
        
        self.loss += tf.reduce_mean(tf.square(self.label - self.y_predict))
        self.params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(1e-3)
        #self.train_op = optimizer.minimize(self.loss)
        grad_var = optimizer.compute_gradients(loss = self.loss, var_list = self.params, aggregation_method = 2)
        self.train_op = optimizer.apply_gradients(grad_var)
        
        
    def en_RNN(self,input_x):
        
        with tf.variable_scope('fw_lstm') as scope:
            cell = self.encode_cell  ## this step don't create variable 
            out, states = tf.nn.dynamic_rnn(cell, input_x, self.seq_len, dtype = tf.float32) ## this step does create variables
            
        tmp = tf.tile(states[1],[1,self.time_step])
        tmp = tf.reshape(tmp,[self.batch_size, self.time_step, self.n_hidden])
        
        concat = tf.concat([out,tmp],axis = 2)
        return concat ## shape shoule be (b,T,2*n_hidden)
    
    def de_RNN(self,h):
        
        with tf.variable_scope('dec_lstm') as scope:
            cell = self.decode_cell
            d,s = tf.nn.dynamic_rnn(cell, h, self.seq_len, dtype = tf.float32)
        tmp = tf.tile(s[1],[1,self.time_step])
        tmp = tf.reshape(tmp,[self.batch_size, self.time_step, self.d_hidden])
        concat = tf.concat([d,tmp],axis = 2)
        return d,concat
        
    def attention(self, out, series_k, scope = None):
   
        with tf.variable_scope('encoder') as scope:
            try:
                mean = 0.0
                stddev = 1.0/(self.n_hidden*self.time_step)
                We = tf.get_variable(name = 'We', dtype=tf.float32,shape = [self.time_step, 2*self.n_hidden],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
                Ve = tf.get_variable(name = 'Ve',dtype=tf.float32,shape = [1,self.time_step],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
            except ValueError:
                scope.reuse_variables()
                We = tf.get_variable('We')
                Ve = tf.get_variable('Ve')       
        W_e = tf.tile(tf.expand_dims(We,0),[self.batch_size,1,1])  # b*T*2d
        brcast = tf.nn.tanh(tf.matmul(W_e,out) + series_k) # b,T,T + b,T,1 = b, T, T
        V_e = tf.tile(tf.expand_dims(Ve,0),[self.batch_size,1,1]) # b,1,T
        
        return tf.matmul(V_e,brcast) # b,1,T
    
    def de_attention(self,out,series_k,scope = None):
        
        with tf.variable_scope('decoder') as scope:
            try:
                mean = 0.0
                stddev = 1.0/(self.d_hidden*self.time_step)
                Wd = tf.get_variable(name = 'Wd', dtype=tf.float32,shape = [self.n_hidden, 2*self.d_hidden],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
                Vd = tf.get_variable(name = 'Vd',dtype=tf.float32,shape = [1,self.n_hidden],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
            except ValueError:
                scope.reuse_variables()
                Wd = tf.get_variable('Wd')
                Vd = tf.get_variable('Vd') 
        W_d = tf.tile(tf.expand_dims(Wd,0),[self.batch_size,1,1])
        brcast = tf.nn.tanh(tf.matmul(W_d,out) + series_k) # b,d,2q * b,2q*T = b,d,T
        #return brcast
        V_d = tf.tile(tf.expand_dims(Vd,0),[self.batch_size,1,1]) # b,1,d
        
        return tf.matmul(V_d,brcast) # b,1,d * b,d,T = b,1,T
    
    def predict(self,x_test,y_test,sess):
        
        train_seq_len =  np.ones(self.batch_size) * self.time_step
        feed = {model.input_x: x_test, 
                model.seq_len: train_seq_len,
                model.input_y: y_test}
        y_hat = sess.run(self.y_predict,feed_dict = feed)
        return y_hat


# In[52]:


batch_size = 256
train_batch_num = int(len(df1)*0.8/batch_size) ## 0.8 of traininng data, 20% for testing
df_test = df1[(train_batch_num*batch_size):]
df1 = df1[:(train_batch_num*batch_size)+65]


# In[87]:


batch_size = 256
INPUT_DIM = 6 # six input feature
time_step = 48 # 12 consecutive data to predict next one
n_hidden = 128 # encoder dim
d_hidden = 64 # decoder dim

## o_hidden = 16 default

current_episode = 0
total_episodes = 100
steps = int(len(df1)/batch_size)

tf.reset_default_graph()

model = ts_prediction(input_dim = 6, time_step = time_step, n_hidden= n_hidden, d_hidden = d_hidden, batch_size = batch_size)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

while current_episode<=total_episodes:
    cumulative_loss = 0.0
    for t in range(steps):
        
        x,y,labels = get_batch(df1,batch_size = batch_size, T = time_step, step = t)
        
        train_seq_len =  np.ones(batch_size) * time_step
        feed = {model.input_x: x, 
                model.seq_len: train_seq_len,
                model.input_y: y,
                model.label: labels}
        loss,_ = sess.run([model.loss,model.train_op],feed_dict = feed)
        cumulative_loss += loss
        if t%10==0:
            print "current_episode %i, steps %i, losses are %f" % ( current_episode, t, cumulative_loss)
            cumulative_loss = 0.0
    current_episode+=1
    #loss = sess.run(model.out,feed_dict = feed)


# In[53]:


x_test,y_test,labels_test = get_batch(df1[-150:],batch_size = batch_size, T = time_step, step = 0) 


# In[34]:


steps_test = int(len(df_test)/batch_size)
test_loss = 0.0
for t in range(steps_test-1):

    x_test,y_test,labels_test = get_batch(df_test,batch_size = batch_size, T = time_step, step = t) 
    
    y_hat = model.predict(x_test,y_test,sess) - labels_test
    test_loss += np.mean(np.square(y_hat))

print "the mean squared error for test data are %f " % (test_loss*1.0/steps_test)
#     train_seq_len =  np.ones(batch_size) * time_step
#     feed = {model.input_x: x, 
#             model.seq_len: train_seq_len,
#             model.input_y: y,
#             model.label: labels}
#     loss,_ = sess.run([model.loss,model.train_op],feed_dict = feed)


# In[38]:


## batch_size 256
steps_test = int(len(df_test)/batch_size)
test_loss = 0.0
for t in range(steps_test-1):

    x_test,y_test,labels_test = get_batch(df_test,batch_size = batch_size, T = time_step, step = t) 
    
    y_hat = model.predict(x_test,y_test,sess) - labels_test
    test_loss += np.mean(np.square(y_hat))

print "the mean squared error for test data are %f " % (test_loss*1.0/steps_test)
#     train_seq_len =  np.ones(batch_size) * time_step
#     feed = {model.input_x: x, 
#             model.seq_len: train_seq_len,
#             model.input_y: y,
#             model.label: labels}
#     loss,_ = sess.run([model.loss,model.train_op],feed_dict = feed)


# ### Model Version 2
# + Change of decode RNN, instead of 2 decoding lstm, but decoder share one lstm  $f_{2}$

# In[21]:


class ts_prediction(object):
    
    def __init__(self, input_dim, time_step, n_hidden, d_hidden, batch_size):

        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.d_hidden = d_hidden
        #self.o_hidden = 16
        
        self.input_dim = input_dim
        self.time_step = time_step
        
        self.seq_len = tf.placeholder(tf.int32,[None])
        self.input_x = tf.placeholder(dtype = tf.float32, shape = [None, None, input_dim]) # b,T,d_in
        self.input_y = tf.placeholder(dtype = tf.float32,shape = [None,self.time_step]) # b,T
        self.label = tf.placeholder(dtype = tf.float32) #b,1
        
        self.encode_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        self.decode_cell = tf.contrib.rnn.LSTMCell(self.d_hidden, forget_bias=1.0, state_is_tuple=True)
        #self.output_cell = tf.contrib.rnn.LSTMCell(self.o_hidden, forget_bias=1.0, state_is_tuple=True)
        
        self.loss = tf.constant(0.0)
            ## ===========  build the model =========== ##
            
            ## ==== encoder ===== ## 
        h_encode, c_state = self.en_RNN(self.input_x)
        c_expand = tf.tile(tf.expand_dims(c_state[1],1),[1,self.time_step,1])
        fw_lstm = tf.concat([h_encode,c_expand],axis = 2) # b,T,2n
        stddev = 1.0/(self.n_hidden*self.time_step)
        Ue = tf.get_variable(name= 'Ue',dtype = tf.float32,
                             initializer = tf.truncated_normal(mean = 0.0, stddev = stddev,shape = [self.time_step,self.time_step]))
        ## (b,d,T) * (b,T,T)  = (b,d,T)
        brcast_UX = tf.matmul(tf.transpose(self.input_x,[0,2,1]),tf.tile(tf.expand_dims(Ue,0),[self.batch_size,1,1]))
        e_list = []
        for k in range(self.input_dim):
            feature_k = brcast_UX[:,k,:] 
            e_k = self.en_attention(fw_lstm,feature_k) # b,T 
            e_list.append(e_k)
        e_mat = tf.concat(e_list,axis = 2)
        alpha_mat = tf.nn.softmax(e_mat) #b,T,d_in
        encode_input = tf.multiply(self.input_x,alpha_mat)
        h_t, c_t = self.en_RNN(encode_input, scopes = 'fw_lstm')
        
        
        h_decode, d_state = self.de_RNN(tf.expand_dims(self.input_y,-1))
        
        d_expand = tf.tile(tf.expand_dims(d_state[1],1),[1,self.time_step,1])
        dec_lstm = tf.concat([h_decode,d_expand],axis = 2) # b,T,2*d_hidden

        Ud = tf.get_variable(name = 'Ud', dtype = tf.float32,
                             initializer = tf.truncated_normal(mean = 0.0, stddev = stddev, shape = [self.n_hidden, self.n_hidden]))
        
        brcast_UDX = tf.matmul(h_t,tf.tile(tf.expand_dims(Ud,0),[self.batch_size,1,1])) # b,T,n_hidden
        
        l_list = []
        for i in range(self.time_step):
            feature_i = brcast_UDX[:,i,:]
            l_i = self.dec_attention(dec_lstm,feature_i)
            l_list.append(l_i)
        l_mat = tf.concat(l_list,axis = 2)
        beta_mat = tf.nn.softmax(l_mat, dim = 1)
        context_list = []
        h_tmp = tf.transpose(h_t,[0,2,1])

        for t in range(self.time_step):
            beta_t = tf.reshape(beta_mat[:,t,:],[self.batch_size,1,self.time_step])
            c_t = tf.reduce_sum(tf.multiply(h_tmp,beta_t),2) # b,T,T -> b,T,1
            context_list.append(c_t)
        c_context = tf.stack(context_list,axis = 2) # b,n_hidden,T
        # b,T,1 b,T,n_hidden -> b,T,n_hidden+1
        c_concat = tf.concat([tf.expand_dims(self.input_y,-1),tf.transpose(c_context,[0,2,1])], axis = 2)
        W_hat = tf.get_variable(name = 'W_hat', dtype = tf.float32,
                                initializer = tf.truncated_normal(mean = 0.0, stddev = stddev,shape = [self.n_hidden+1,1]))
        y_encode = tf.matmul(c_concat,tf.tile(tf.expand_dims(W_hat,0),[self.batch_size,1,1]))
        
        
        h_out, d_out = self.de_RNN(y_encode)
        
        last_concat = tf.expand_dims(tf.concat([h_out[:,-1,:],d_out[-1]],axis = 1),1)
        Wy = tf.get_variable(name = 'Wy', dtype = tf.float32,initializer = tf.truncated_normal(mean = 0.0, stddev = stddev,shape = [self.n_hidden+self.d_hidden,1]))
        W_y = tf.tile(tf.expand_dims(Wy,0),[self.batch_size,1,1])
        self.y_predict = tf.squeeze(tf.matmul(last_concat,W_y))
        self.loss += tf.reduce_mean(tf.square(self.label - self.y_predict)) # reduce_mean: avg of batch loss
        self.params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(1e-3)
        #self.train_op = optimizer.minimize(self.loss)
        grad_var = optimizer.compute_gradients(loss = self.loss, var_list = self.params, aggregation_method = 2)
        self.train_op = optimizer.apply_gradients(grad_var)
        
        
    def en_RNN(self, input_x, scopes = 'fw_lstm'):
        '''
        input_x: b, T, d_in
        
        output: h:       seqence of output state b,T,n_hidden
                state:   final state b,n_hidden
                
        '''
        with tf.variable_scope('fw_lstm' or scopes) as scope:
            try:
                h,state = tf.nn.dynamic_rnn(
                    cell = self.encode_cell, inputs = input_x,
                    sequence_length = self.seq_len,
                    dtype = tf.float32, scope = 'fw_lstm')
                
            except ValueError:
                scope.reuse_variables()
                h,state = tf.nn.dynamic_rnn(
                    cell = self.encode_cell, inputs = input_x,
                    sequence_length = self.seq_len,
                    dtype = tf.float32, scope = scopes)
        
        return [h,state]
    
    def de_RNN(self,input_y, scopes = 'de_lstm'):
        
        with tf.variable_scope('dec_lstm') as scope:
            try:
                h,state = tf.nn.dynamic_rnn(
                    cell = self.decode_cell, inputs = input_y,
                    sequence_length = self.seq_len,
                    dtype = tf.float32, scope = 'de_lstm')
                
            except ValueError:
                scope.reuse_variables()
                h,state = tf.nn.dynamic_rnn(
                    cell = self.decode_cell, inputs = input_y,
                    sequence_length = self.seq_len,
                    dtype = tf.float32, scope = scopes)
        
        return [h,state]
    
    def en_attention(self,fw_lstm,feature_k):
        '''
        fw_lstm: b,T,2n
        feature_k: row k from brcast_UX, b,T
        
        return: b,T
        '''
        with tf.variable_scope('encoder') as scope:
            try:
                mean = 0.0
                stddev = 1.0/(self.n_hidden*self.time_step)
                We = tf.get_variable(name = 'We', dtype=tf.float32,shape = [self.time_step, 2*self.n_hidden],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
                Ve = tf.get_variable(name = 'Ve',dtype=tf.float32,shape = [self.time_step,1],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
            except ValueError:
                scope.reuse_variables()
                We = tf.get_variable('We')
                Ve = tf.get_variable('Ve')   
            # (b,T,2n) (b,2n,T)
        W_e = tf.transpose(tf.tile(tf.expand_dims(We,0),[self.batch_size,1,1]),[0,2,1]) # b,2n,T
        mlp = tf.nn.tanh(tf.matmul(fw_lstm,W_e) + tf.reshape(feature_k,[self.batch_size,1,self.time_step])) #b,T,T + b,1,T = b,T,T
        V_e = tf.tile(tf.expand_dims(Ve,0),[self.batch_size,1,1])
        return  tf.matmul(mlp,V_e)
            
    def dec_attention(self, dec_lstm, feature_t, scopes = None):
        '''
        dec_lstm: b,T,2*d_hidden
        feature_k: row k from brcast_UX, b,T
        
        return: b,T
        '''
        with tf.variable_scope('decoder' or scopes) as scope:
            try:
                mean = 0.0
                stddev = 1.0/(self.n_hidden*self.time_step)
                Wd = tf.get_variable(name = 'Wd', dtype=tf.float32, shape = [self.n_hidden, 2*self.d_hidden],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
                Vd = tf.get_variable(name = 'Vd', dtype=tf.float32, shape = [self.n_hidden,1],
                                     initializer=tf.truncated_normal_initializer(mean,stddev))
            except ValueError:
                scope.reuse_variables()
                Wd = tf.get_variable('Wd')
                Vd = tf.get_variable('Vd')   
        # (b,T,2*d_hidden) (b,2*d_hidden,T)
        W_d = tf.transpose(tf.tile(tf.expand_dims(Wd,0),[self.batch_size,1,1]),[0,2,1]) # b,2*d_hidden,n_hidden
        # (b,T,2*d_hidden) * (b,2*d_hidden,n_hidden) -> b,T,n_hidden
        mlp = tf.nn.tanh(tf.matmul(dec_lstm,W_d) + tf.reshape(feature_t,[self.batch_size,1,self.n_hidden])) #b,T,n_hidden + b,1,n_hidden = b,T,n_hidden
        V_d = tf.tile(tf.expand_dims(Vd,0),[self.batch_size,1,1])
        return  tf.matmul(mlp,V_d) #b,T,1
    
    def predict(self,x_test,y_test,sess):
        
        train_seq_len =  np.ones(self.batch_size) * self.time_step
        feed = {model.input_x: x_test, 
                model.seq_len: train_seq_len,
                model.input_y: y_test}
        y_hat = sess.run(self.y_predict,feed_dict = feed)
        return y_hat


# In[23]:


batch_size = 256
train_batch_num = int(len(df1)*0.8/batch_size) ## 0.8 of traininng data, 20% for testing

df_test = df1[(train_batch_num*batch_size):]
df1 = df1[:(train_batch_num*batch_size)+65]

print 'Training data %i' % len(df1), "testing data %i" % len(df_test)


# In[25]:


batch_size = 256
INPUT_DIM = 6 # six input feature
time_step = 48 # 12 consecutive data to predict next one
n_hidden = 64 # encoder dim
d_hidden = 64 # decoder dim
## o_hidden = 16 default

current_episode = 0
total_episodes = 50
steps = int(len(df1)/batch_size)

tf.reset_default_graph()

model = ts_prediction(input_dim = 6, time_step = time_step, n_hidden= n_hidden, d_hidden = d_hidden, batch_size = batch_size)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

while current_episode<=total_episodes:
    cumulative_loss = 0.0
#     index = range(train_batch_num*batch_size)
#     np.random.shuffle(index)
#     df1 = df1.iloc[index]
    
    for t in range(steps):
        ## shuffle the batch
        index = range(x.shape[0])
        np.random.shuffle(index)
        x,y,labels = get_batch(df1,batch_size = batch_size, T = time_step, step = t)
        train_seq_len =  np.ones(batch_size) * time_step
        feed = {model.input_x: x, 
                model.seq_len: train_seq_len,
                model.input_y: y,
                model.label: labels}
        loss,_ = sess.run([model.loss,model.train_op],feed_dict = feed)
        cumulative_loss += loss
        if t%10==0:
            print "current_episode %i, steps %i, losses are %f" % ( current_episode, t, cumulative_loss)
            cumulative_loss = 0.0
    current_episode+=1


# In[62]:


steps_test = int(len(df_test)/batch_size)
test_loss = 0.0
y_hat_arr = np.empty(shape = [0])
y_labels_arr = np.empty(shape = [0])

for t in range(steps_test-1):

    x_test,y_test,labels_test = get_batch(df_test,batch_size = batch_size, T = time_step, step = t, train = False)   
    y_hat = model.predict(x_test,y_test,sess) 
    y_hat_arr = np.concatenate([y_hat_arr,np.array(y_hat)])
    y_labels_arr = np.concatenate([y_labels_arr,np.array(labels_test)])
    test_loss += np.mean(np.square(y_hat - labels_test))

print "the mean squared error for test data are %f " % (test_loss*1.0/steps_test)


# In[77]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plot_num = 256

for i in range(steps_test-1):
    start_idx = i*plot_num
    end_idx = (i+1)*plot_num
    plt.figure()
    plt.plot(range(plot_num),y_hat_arr[start_idx:end_idx])
    plt.plot(range(plot_num),y_labels_arr[start_idx:end_idx])
    plt.draw()
    plt.savefig('./time_series/range_%i.png' % i)


# In[81]:


print 'traninig data point %i' % len(df1)
print 'testing data point %i' % len(df_test)
print "the mean squared error for test data are %f " % (test_loss*1.0/steps_test)

