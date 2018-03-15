
import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
# Layer params:   Filts K  Padding  Name     BatchNorm?
layer_params = [ [  64, 3, 'same',  'conv1', False], 
                 [  64, 3, 'same',  'conv2', True], # pool
                 [ 128, 3, 'same',  'conv3', False], 
                 [ 128, 3, 'same',  'conv4', True], # hpool
                 [ 256, 3, 'same',  'conv5', False],
                 [ 256, 3, 'same',  'conv6', True], # hpool
                 [ 256, 3, 'same',  'conv7', False], 
                 [ 256, 3, 'same',  'conv8', True]] # hpool 3

rnn_size = 2**8
dropout_rate = 0.5

def conv_layer(bottom, params, training ):
    """Build a convolutional layer using entry from layer_params)"""

    batch_norm = params[4] # Boolean

    if batch_norm:
        activation=None
    else:
        activation=tf.nn.relu

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    top = tf.layers.conv1d(bottom, 
                           filters=params[0],
                           kernel_size=params[1],
                           padding=params[2],
                           activation=activation,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           name=params[3])
    if batch_norm:
        top = norm_layer( top, training, params[3]+'/batch_norm' )
        top = tf.nn.relu( top, name=params[3]+'/relu' )

    return top

def pool_layer( bottom, wpool, padding, name ):
    """Short function to build a pooling layer with less syntax"""
    top = tf.layers.max_pooling1d( bottom, 2, wpool, 
                                   padding=padding, 
                                   name=name)
    return top

def norm_layer( bottom, training, name):
    """Short function to build a batch normalization layer with less syntax"""
    top = tf.layers.batch_normalization( bottom, axis=2, # channels last,
                                         training=training,
                                         name=name )
    return top


def convnet_layers(inputs, widths, mode):
    """Build convolutional network layers attached to the given input tensor"""
    input('convnet_layers')
    training = (mode == learn.ModeKeys.TRAIN)

    # inputs should have shape [ ?, 32, ?, 1 ]
    # h,w
    print(inputs)
    print(inputs.dtype)
    #n0 = tf.shape(inputs)[0]
    #n1 = tf.shape(inputs)[1]
    #n2 = tf.shape(inputs)[2]

    #if inputs.dtype != tf.float32:
    #  plt.imshow(inputs[0,:,:])
    #  plt.show()


    inputs_ = tf.reshape(inputs,[-1,32,625])
    #inputs_ = tf.squeeze(inputs,axis=3)
    print(inputs_)
    print(inputs_.get_shape())
    input('inputs')
    '''
    #with tf.variable_scope("convnet"):
        
        #(batch,32,625) -> (batch,)
        # (batch, 128, 9) --> (batch, 64, 18)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1, 
                           padding='same', activation = tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
        print(max_pool_1)
        input('max_pool1')
        # (batch, 64, 18) --> (batch, 32, 36)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
        print(max_pool_2)
        input('max_pool2')
        # (batch, 32, 36) --> (batch, 16, 72)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
        print(max_poo_l3)
        input('max_pool3')
        # (batch, 16, 72) --> (batch, 8, 144)
        conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

        print(max_pool_4)
        input('max_pool_4')
        flat = tf.reshape(max_pool_4, (-1, 2*144))
        flat = tf.nn.dropout(flat, keep_prob=0.5)

        # Predictions
        logits = tf.layers.dense(flat, 10)
        features = logits
        sequence_length = 0
    '''

    with tf.variable_scope("convnet"):

        input('convnet_layers22')
        conv1 = conv_layer(inputs_, layer_params[0], training ) # 30,30
        input('convnet_layers23')
        conv2 = conv_layer( conv1, layer_params[1], training ) # 30,30
        input('convnet_layers24')
        pool2 = pool_layer( conv2, 1, 'valid', 'pool2')        # 15,15
        input('convnet_layers25')
        conv3 = conv_layer( pool2, layer_params[2], training ) # 15,15
        input('convnet_layers26')
        conv4 = conv_layer( conv3, layer_params[3], training ) # 15,15
        pool4 = pool_layer( conv4, 1, 'valid', 'pool4' )       # 7,14
        conv5 = conv_layer( pool4, layer_params[4], training ) # 7,14
        conv6 = conv_layer( conv5, layer_params[5], training ) # 7,14
        pool6 = pool_layer( conv6, 1, 'valid', 'pool6')        # 3,13
        conv7 = conv_layer( pool6, layer_params[6], training ) # 3,13
        conv8 = conv_layer( conv7, layer_params[7], training ) # 3,13
        pool8 = tf.layers.max_pooling1d( conv8, 2, 1,  # [3,1], [3,1], 
                                   padding='valid', name='pool8') # 1,13

        ## ADD DROPOUT

        #features = tf.squeeze(pool8, [2], name='features') # squeeze row dim
        flat = tf.reshape(pool8, (-1, 28*256))
        flat = tf.nn.dropout(flat, keep_prob=0.5)
        logits = tf.layers.dense(flat, 10)
        features = logits

        #tf.Print(features)
        #tf.summary.scalar('features',features)
        #input('convnet_layers2')
        '''
        kernel_sizes = [ params[1] for params in layer_params]
        
        # Calculate resulting sequence length from original image widths
        #print(widths)
        
        #input('width')
        conv1_trim = tf.constant( 2 * (kernel_sizes[0] // 2),
                                  dtype=tf.int32,
                                  name='conv1_trim')
        one = tf.constant(1, dtype=tf.int32, name='one')
        two = tf.constant(2, dtype=tf.int32, name='two')
        after_conv1 = tf.subtract( widths, conv1_trim)
        after_pool2 = tf.floor_div( after_conv1, two )
        after_pool4 = tf.subtract(after_pool2, one)
        sequence_length = tf.reshape(after_pool4,[-1], name='seq_len') # Vectorize
        print(sequence_length)
        input('asd')
        input('convnet_layers3')
        
        #sequence_length = 0
        '''

    return features,conv1,inputs_

def rnn_layer(bottom_sequence,sequence_length,rnn_size,scope):
    """Build bidirectional (concatenated output) RNN layer"""

    weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    
    # Default activation is tanh
    cell_fw = tf.contrib.rnn.LSTMCell( rnn_size, 
                                       initializer=weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell( rnn_size, 
                                       initializer=weight_initializer)
    # Include?
    #input('rnn_layer1')
    #cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw, 
    #                                         input_keep_prob=dropout_rate )
    #cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw, 
    #                                         input_keep_prob=dropout_rate )
    
    rnn_output,_ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, bottom_sequence,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.float32,
        scope=scope)
    input('rnn_layer2')
    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ paddedSeqLen batchSize 2*rnn_size]
    rnn_output_stack = tf.concat(rnn_output,2,name='output_stack')
    
    return rnn_output_stack


def rnn_layers(features, sequence_length, num_classes):
    """Build a stack of RNN layers from input features"""

    # Input features is [batchSize paddedSeqLen numFeatures]
    logit_activation = tf.nn.relu
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    input('rnn_layers1')
    with tf.variable_scope("rnn"):
        # Transpose to time-major order for efficiency
        rnn_sequence = tf.transpose(features, perm=[1, 0, 2], name='time_major')
        rnn1 = rnn_layer(rnn_sequence, sequence_length, rnn_size, 'bdrnn1')
        #rnn2 = rnn_layer(rnn1, sequence_length, rnn_size, 'bdrnn2')
        rnn_logits = tf.layers.dense( rnn1, num_classes+1, 
                                      activation=logit_activation,
                                      kernel_initializer=weight_initializer,
                                      bias_initializer=bias_initializer,
                                      name='logits')
        return rnn_logits
    

def ctc_loss_layer(rnn_logits, sequence_labels, sequence_length):
    """Build CTC Loss layer for training"""
    loss = tf.nn.ctc_loss( sequence_labels, rnn_logits, sequence_length,
                           time_major=True )
    total_loss = tf.reduce_mean(loss)
    return total_loss

#NEW
def crwl_loss_layer(rnn_logits,lables):
  loss = tf.nn.softmax_cross_entropy_with_logits(logits=rnn_logits, labels=lables)
  total_loss = tf.reduce_mean(loss)
  return total_loss
