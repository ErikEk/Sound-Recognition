
import os
import tensorflow as tf
import math
import pandas as pd
import librosa as lb
import sys
import numpy as np
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import soundfile as sf
from glob import glob

"""Each record within the TFRecord file is a serialized Example proto. 
The Example proto contains the following fields:
  image/encoded: string containing JPEG encoded grayscale image
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/filename: string containing the basename of the image file
  image/labels: list containing the sequence labels for the image text
  image/text: string specifying the human-readable version of the text
"""

# The list (well, string) of valid output characters
# If any example contains a character not found here, an error will result
# from the calls to .index in the decoder below
#NEW

out_charset="0123456789"
label_features = {"blues":[1,0,0,0,0,0,0,0,0,0],\
                  "classical":[0,1,0,0,0,0,0,0,0,0],\
                  "country":[0,0,1,0,0,0,0,0,0,0],\
                  "hiphop":[0,0,0,1,0,0,0,0,0,0],\
                  "jazz":[0,0,0,0,1,0,0,0,0,0],\
                  "metal":[0,0,0,0,0,1,0,0,0,0],\
                  "reggae":[0,0,0,0,0,0,1,0,0,0],\
                  "rock":[0,0,0,0,0,0,0,1,0,0],\
                  "pop":[0,0,0,0,0,0,0,0,1,0],\
                  "disco":[0,0,0,0,0,0,0,0,0,1]}
#OLD
#out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

jpeg_data = tf.placeholder(dtype=tf.string)
jpeg_decoder = tf.image.decode_jpeg(jpeg_data,channels=1)

kernel_sizes = [5,5,3,3,3,3] # CNN kernels for image reduction

# Minimum allowable width of image after CNN processing
min_width = 20

def calc_seq_len(image_width):
    """Calculate sequence length of given image after CNN processing"""
    
    conv1_trim =  2 * (kernel_sizes[0] // 2)
    fc6_trim = 2*(kernel_sizes[5] // 2)
    
    after_conv1 = image_width - conv1_trim 
    after_pool1 = after_conv1 // 2
    after_pool2 = after_pool1 // 2
    after_pool4 = after_pool2 - 1 # max without stride
    after_fc6 =  after_pool4 - fc6_trim
    seq_len = 2*after_fc6
    return seq_len

seq_lens = [calc_seq_len(w) for w in range(8192)]

def gen_data(input_base_dir, image_list_filename, output_filebase, 
             num_shards=10,start_shard=0):
    """ Generate several shards worth of TFRecord data """
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    sess = tf.Session(config=session_config)

    #OLD
    '''
    dir_list = glob(input_base_dir+'/*/')
    ind = 0

    audio_filenames = ['' for x in range(10*100)]
    for path, subdirs, files in os.walk(input_base_dir+'/'):
        for name in files:
            audio_filenames[ind] = os.path.join(path, name)
            ind+=1
            print(os.path.join(path, name))
    '''

    image_filenames = get_image_filenames(os.path.join(input_base_dir,
                                                       image_list_filename))
    image_filenames = [input_base_dir + '/' + s for s in image_filenames]
    #image_filenames = image_filenames + input_base_dir
    #print(image_filenames)
    #print(type(image_filenames))
    #input('image_filenames')
    #image_filenames = audio_filenames

    #full_csv['slice_file_name'][full_csv.shape[0]-1]
    #len(image_filenames)
    #input('stop')

    num_digits = math.ceil( math.log10( num_shards - 1 ))
    shard_format = '%0'+ ('%d'%num_digits) + 'd' # Use appropriate # leading zeros
    images_per_shard = int(math.ceil( len(image_filenames) / float(num_shards) ))
    
    for i in range(start_shard,num_shards): # make sure it just is num_shards-1 shards
        start = i*images_per_shard
        end   = (i+1)*images_per_shard
        out_filename = output_filebase+'-'+(shard_format % i)+'.tfrecord'
        if os.path.isfile(out_filename): # Don't recreate data if restarting
            print('found: ',out_filename)
            continue
        print(str(i),'of',str(num_shards),'[',str(start),':',str(end),']',out_filename)
        gen_shard(sess, input_base_dir, image_filenames[start:end], out_filename)
    # Clean up writing last shard
    start = num_shards*images_per_shard
    out_filename = output_filebase+'-'+(shard_format % num_shards)+'.tfrecord'
    print(str(i),'of',str(num_shards),'[',str(start),':]',out_filename)
    gen_shard(sess, input_base_dir, image_filenames[start:], out_filename)

    sess.close()
Fs         = 22050
N_FFT      = 2048
N_MELS     = 32
N_OVERLAP  = 1024

def get_duration(path, bitdepth, samplerate,channels):
    return os.path.getsize(path)/bitdepth*8/samplerate/channels # 1 channels

def log_scale_melspectrogram(sess,path, plot=False):

    bitdepth = 16 # GUESS

    signal, sr = lb.load(path, sr=Fs, mono=True)

    n_sample = signal.shape[0]


    # Get duration
    DURA = get_duration(path,bitdepth,Fs,1)# 1 channels

    


    print(DURA)
    print(path)
    print('n_sample:',n_sample)
    print(30*Fs)
    print(signal)
    print(signal.shape)
    #input('tttt')
    signal = signal[0:Fs*29] # Capture 29 sec

    #input('tttt2')

    #input('duration')
    #n_sample_fit = int(DURA*Fs)
    #print('n_sample_fit: ',n_sample_fit)
    

    '''
    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
    elif n_sample > n_sample_fit:
        #print((n_sample-n_sample_fit)/2)
        signal = signal[(n_sample-n_sample_fit)//2:(n_sample+n_sample_fit)//2] # HAR LAGT TILL int()
    '''
    

    ## Not Sure
    #

    #print(len(signal))
    #print(Fs)
    melspect = lb.logamplitude(lb.feature.melspectrogram(y=signal, sr=Fs,hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS )**2, ref_power=1.0)
    # INTE SÄKER HÄR
    #print(melspect.shape)
    melspect = melspect[np.newaxis, :]
    #print(melspect.shape)
    #input('ba')
    if plot:

        #print(melspect.shape)
        #plt.imshow(melspect)
        plt.imshow(melspect.reshape((melspect.shape[1],melspect.shape[2])))
        plt.show()

    #print(melspect.reshape((melspect.shape[1],melspect.shape[2])).shape)
    #input('retrunb')
    return [melspect.reshape((melspect.shape[1],melspect.shape[2])),melspect.shape[1],melspect.shape[2]]

resave_waves_as_jpgs = False

def gen_shard(sess, input_base_dir, image_filenames, output_filename):
    """Create a TFRecord file from a list of image filenames"""
    writer = tf.python_io.TFRecordWriter(output_filename)
    
    for filename in image_filenames:
        #path_filename = os.path.join(input_base_dir,filename)
        path_filename = filename
        #print(filename)
        #input('sss')
        filename = filename.split('/')[4]
        #print(filename)
        #input('sss')
        new_folder = '../../genres_spec/'
        #print('wave file: ', path_filename)
        
        #if os.stat(path_filename).st_size == 0:
        #    print('SKIPPING',filename)
        #    continue
        #try:
        #print(path_filename)
        #input('path_filename')
        image_filename = new_folder+path_filename.split('/')[3]+'/'+filename[:-3]+'.jpg'
        print('looking for file: ',image_filename)
        #input('sss')
        #If ty
        if os.path.isfile(image_filename) and resave_waves_as_jpgs==False:
            print('skipping jpg file', image_filename)
        else:
            #print('saving: ',image_filename)
            #input('saving')
            image_data_temp, height, width = log_scale_melspectrogram(sess,path_filename,False)
            print('save',image_filename)
            scipy.misc.imsave(image_filename, image_data_temp)

        image_data,height,width = get_image(sess,image_filename)
        #image_data,height,width = get_cropped_image(sess,image_filename)

        #image_data = bytearray(image_data_temp)
        #print(type(image_data))
        #input('type')

        #de_image = sess.run(decoder,feed_dict={j_data:image_data})

        #for i in range(0,3):
        #croped_image = sess.run(cropper,feed_dict={jp_data: de_image})
        #width = 100 # 2048 # 3000 total

        #split_image(20)
        text,labels = get_text_and_labels(filename)
        #input('1')
        if is_writable(width,text):
            #input('2')
            example = make_example(filename, image_data, labels, text, 
                                   height, width)
            #input('3')
            writer.write(example.SerializeToString())
        else:
            print('SKIPPING',filename)
            #input('skipping')
        #except:
            # Some files have bogus payloads, catch and note the error, moving on
            #print('ERROR',filename)
    writer.close()


def slipt_image(mn=20):
    return 0

def get_image_filenames(image_list_filename):
    """ Given input file, generate a list of relative filenames"""
    filenames = []
    with open(image_list_filename) as f:
        for line in f:
            # Carve out the ground truth string and file path from lines like:
            # ./2697/6/466_MONIKER_49537.jpg 49537
            filename = line.split(' ',1)[0][2:-1]#line.split(' ',1)[0][2:] # split off "./" and number
            print('file: ',filename)
            filenames.append(filename)
    return filenames

def get_image(sess,filename):
    """Given path to an image file, load its data and size"""
    with tf.gfile.FastGFile(filename, 'rb') as f: # added 'b' python3
        image_data = f.read()

    image = sess.run(jpeg_decoder,feed_dict={jpeg_data: image_data})
    height = image.shape[0]
    width = image.shape[1]
    print('width: ',width)
    if width!=625:
        print(filename)
        input('not 625')



    return image_data, height, width

def get_cropped_image(sess,filename):
    """Given path to an image file, load its data and size"""
    with tf.gfile.FastGFile(filename, 'rb') as f: # added 'b' python3
        image_data = f.read()
    image = sess.run(jpeg_decoder,feed_dict={jpeg_data: image_data})

    croped_image = sess.run(cropper,feed_dict={jp_data: image})

    #image = sess.run(jpeg_decoder,feed_dict={jpeg_data: croped_image})
    height = croped_image.shape[0]
    width = croped_image.shape[1]

    return croped_image, height, width

def is_writable(image_width,text):
    """Determine whether the CNN-processed image is longer than the string"""
    return (image_width > min_width) and (len(text) <= seq_lens[image_width])
    
def get_text_and_labels(filename):
    """ Extract the human-readable text and label sequence from image filename"""
    # Ground truth string lines embedded within base filename between underscores
    # 2697/6/466_MONIKER_49537.jpg --> MONIKER
    #NEW
    #print(filename)
    #input('texten som')
    text = os.path.basename(filename).split('.',2)[1]
    #print(text)
    #input('after')
    #OLD
    labels = label_features[filename.split('.')[0]]
    print(labels)
    #input('ssss')
    # Transform string text to sequence of indices using charset, e.g.,
    # MONIKER -> [12, 14, 13, 8, 10, 4, 17]
    #NEW
    #labels = [out_charset.index(c) for c in list(text)]
    #print(text)
    #print(labels)
    #input('abe')
    #OLD 
    #labels = [out_charset.index(c) for c in list(text)]
    return text,labels

def make_example(filename, image_data, labels, text, height, width):
    """Build an Example proto for an example.
    Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_data: string, JPEG encoding of grayscale image
    labels: integer list, identifiers for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

    print('lables',labels)
    print('width:',width)
    print('height:',height)
    print('filename',filename)
    print('text',text)
    print('textlen',len(text))
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data)),
        'image/labels': _int64_feature(labels),
        'image/height': _int64_feature([height]),
        'image/width': _int64_feature([width]),
        'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
        'text/string': _bytes_feature(tf.compat.as_bytes(text)),
        'text/length': _int64_feature([len(text)])
    }))
    return example

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def main(argv=None):
    
    gen_data('../../genres','notations_train.txt','../data/genres/gen')
    #gen_data('../../genres','all','../data/genres/gen')
    #gen_data('../data/images', 'annotation_val.txt',   '../data/val/words')
    #gen_data('../data/images', 'annotation_test.txt',  '../data/test/words')

if __name__ == '__main__':
    main()
