import os
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import mjsynth_genre
import model_genre
import matplotlib.pyplot as plt
import pprint

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output','../data/model_mel',
                          """Directory for event logs and checkpoints""")
tf.app.flags.DEFINE_string('tune_from','',
                          """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('tune_scope','',
                          """Variable scope for training""")

tf.app.flags.DEFINE_integer('batch_size',2**5,
                            """Mini-batch size""")
tf.app.flags.DEFINE_float('learning_rate',1e-4, # 1e-4
                          """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum',0.99,
                          """Optimizer gradient first-order momentum""")
tf.app.flags.DEFINE_float('decay_rate',0.9,#0.9
                          """Learning rate decay base""")
tf.app.flags.DEFINE_float('decay_steps',2**16,
                          """Learning rate decay exponent scale""")
tf.app.flags.DEFINE_float('decay_staircase',False,
                          """Staircase learning rate decay by integer division""")


tf.app.flags.DEFINE_integer('max_num_steps', 2**21,
                            """Number of optimization steps to run""")

tf.app.flags.DEFINE_string('train_device','/gpu:0',
                           """Device for training graph placement""")
tf.app.flags.DEFINE_string('input_device','/gpu:0',
                           """Device for preprocess/batching graph placement""")

tf.app.flags.DEFINE_string('train_path','../data/genres/',
                           """Base directory for training data""")
tf.app.flags.DEFINE_string('filename_pattern','gen-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads',4, # FUNKAR FÖR NU
                          """Number of readers for input data""")
tf.app.flags.DEFINE_integer('width_threshold',None,
                            """Limit of input image width""")
tf.app.flags.DEFINE_integer('length_threshold',None,
                            """Limit of input string length width""")

tf.logging.set_verbosity(tf.logging.INFO)

# Non-configurable parameters
optimizer='Adam'
mode = learn.ModeKeys.TRAIN # 'Configure' training mode for dropout layers

def _get_input():
    """Set up and return image, label, and image width tensors"""

    image,width,label,_,_,_= mjsynth_genre.bucketed_input_pipeline(
        FLAGS.train_path, 
        str.split(FLAGS.filename_pattern,','),
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_input_threads,
        input_device=FLAGS.input_device,
        width_threshold=FLAGS.width_threshold,
        length_threshold=FLAGS.length_threshold)

    return image,width,label

def _get_single_input():
    """Set up and return image, label, and width tensors"""

    image,width,label,length,text,filename=mjsynth_genre.threaded_input_pipeline(
        FLAGS.train_path, 
        str.split(FLAGS.filename_pattern,','),
        batch_size=32,
        num_threads=FLAGS.num_input_threads,
        num_epochs=None,
        batch_device=FLAGS.input_device, 
        preprocess_device=FLAGS.input_device )

    return image,width,label,filename

def _get_training(rnn_logits,label,sequence_length):
    """Set up training ops"""
    with tf.name_scope("train"):

        if FLAGS.tune_scope:
            scope=FLAGS.tune_scope
        else:
            scope="convnet|rnn"

        #print(label)
        #input('label')
        rnn_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope)

        label_den = tf.sparse_tensor_to_dense(label)


        loss = model_genre.crwl_loss_layer(rnn_logits,label_den)


        input('_get_training1')
        # Update batch norm stats [http://stackoverflow.com/questions/43234667]
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        input('_get_training2')
        with tf.control_dependencies(extra_update_ops):

            
            learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate,
                tf.train.get_global_step(),
                FLAGS.decay_steps,
                FLAGS.decay_rate,
                staircase=FLAGS.decay_staircase,
                name='learning_rate')
            
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=FLAGS.momentum)
            
            # Accuracy
            correct_pred = tf.equal(tf.argmax(rnn_logits, 1), tf.argmax(label_den, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
            
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=learning_rate, 
                optimizer=optimizer,
                variables=rnn_vars)
            input('_get_training2')
            tf.summary.scalar( 'learning_rate', learning_rate )
            
            
    return train_op,accuracy#accuracy,loss,optimizer

def _get_session_config():
    """Setup session config to soften device placement"""

    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config

def _get_init_pretrained():
    """Return lambda for reading pretrained initial model"""

    if not FLAGS.tune_from:
        return None
    
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    ckpt_path=FLAGS.tune_from

    init_fn = lambda sess: saver_reader.restore(sess, ckpt_path)

    return init_fn


def main(argv=None):

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        

        image,width,label,filename = _get_single_input()
        print(image)
        input('main1')
        with tf.device(FLAGS.train_device):
            features,conv1,inputs = model_genre.convnet_layers( image, width, mode)
            #logits = model_genre.rnn_layers( features, sequence_length,
            #                           mjsynth_genre.num_classes() )

            sequence_length = 10
            logits = features
            train_op,accuracy  = _get_training(logits,label,sequence_length)
            #accuracy,loss,optimizer


        session_config = _get_session_config()

        summary_op = tf.summary.merge_all()
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 

        sv = tf.train.Supervisor(
            logdir=FLAGS.output,
            init_op=init_op,
            summary_op=summary_op,
            save_summaries_secs=10,
            init_fn=_get_init_pretrained(),
            save_model_secs=30)

        input('main3')
        with sv.managed_session(config=session_config) as sess:
            step = sess.run(global_step)

            
            print(step)
            while step < FLAGS.max_num_steps:
                if sv.should_stop():
                    break

                [step_loss,accuracy_,step]=sess.run([ train_op,accuracy,global_step])

                if step%100==0:
                    print(step_loss)
                    print('accuracy: ',accuracy_)
                #input__,label__,filename__ = sess.run([image,label,filename])
                #print(filename__)
                #print(label__.values)
                #test = np.array(input__)
                #print(test.shape)
                #print('size: ', test.size)
                #plt.imshow(test[0,:,:])
                #plt.show()

                #print(input__)
                #print(type(input__))
                #print(len(input__[0,0]))
                #input('step:')

            sv.saver.save( sess, os.path.join(FLAGS.output,'model.ckpt'),
                           global_step=global_step)

if __name__ == '__main__':
    tf.app.run()

