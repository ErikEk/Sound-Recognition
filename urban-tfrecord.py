
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
             num_shards=100,start_shard=0): # 1000 if urban
    """ Generate several shards worth of TFRecord data """
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    sess = tf.Session(config=session_config)
    #image_filenames = get_image_filenames(os.path.join(input_base_dir,
    #                                                   image_list_filename))

    full_csv = pd.read_csv(image_list_filename,header=0)
    

    filenamelist = 'fold'+full_csv['fold'].astype(str)+\
        '/'+full_csv['slice_file_name']

    #print(filenamelist)
    image_filenames = filenamelist

    #full_csv['slice_file_name'][full_csv.shape[0]-1]
    #len(image_filenames)
    #input('stop')

    num_digits = math.ceil( math.log10( num_shards - 1 ))
    shard_format = '%0'+ ('%d'%num_digits) + 'd' # Use appropriate # leading zeros
    images_per_shard = int(math.ceil( len(image_filenames) / float(num_shards) ))
    
    for i in range(start_shard,num_shards):
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
#Fs         = 44100
#Fs          = 22050
N_FFT      = 521
N_MELS     = 31
N_OVERLAP  = 128

def get_duration(path, bitdepth, samplerate,channels):
    return os.path.getsize(path)/bitdepth*8/samplerate/channels # 2 channels

def log_scale_melspectrogram(sess,path, plot=False):

    #Fs, data = scipy.io.wavfile.read(path)
    data,Fs = sf.read(path)
    soundfile_obj = sf.SoundFile(path)
    bitdepth = -1

    if soundfile_obj.subtype == 'PCM_U8':
        bitdepth = 8
    elif soundfile_obj.subtype == 'PCM_16':
        bitdepth = 16
    elif soundfile_obj.subtype == 'PCM_24':
        bitdepth = 24
    elif soundfile_obj.subtype == 'PCM_32':
        bitdepth = 32

    if bitdepth == -1: # IN CASE OF DIFF
        print(soundfile_obj.subtype)

    # Get number of channels
    channels = soundfile_obj.channels

    #print(bitdepth)
    #print(Fs)
    #print(soundfile_obj.subtype)
    #print(soundfile_obj.channels)
    #input('subtype')
    #print(Fs)
    #input('rate')
    signal, sr = lb.load(path, sr=Fs)
    #print('---',path,sr)
    #size_acumulated += sys.getsizeof(signal)
    #print(convert_size(size_acumulated))
    #input('ss')
    n_sample = signal.shape[0]
    #D = np.abs(librosa.stft(signal))**2
    # Get duration
    DURA = get_duration(path,bitdepth,Fs,channels)

    #print(DURA)

    #input('duration')
    n_sample_fit = int(DURA*Fs)

    #input('testtt')
    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
    elif n_sample > n_sample_fit:
        #print((n_sample-n_sample_fit)/2)
        signal = signal[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)] # HAR LAGT TILL int()

    ## TESTAR UTAN DESSA, KAN BLI FEL!!!!
    #, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS 

    #print(len(signal))
    #print(Fs)
    melspect = lb.logamplitude(lb.feature.melspectrogram(y=signal, sr=Fs,n_mels=N_MELS)**2, ref_power=1.0)
    # INTE SÄKER HÄR
    melspect = melspect[np.newaxis, :]
    if plot:

        #print(melspect.shape)
        #plt.imshow(melspect)
        plt.imshow(melspect.reshape((melspect.shape[1],melspect.shape[2])))
        plt.show()
    #input('size')

    # Close file
    soundfile_obj.close()

    return [melspect.reshape((melspect.shape[1],melspect.shape[2])),melspect.shape[1],melspect.shape[2]]

resave_waves_as_jpgs = False

def gen_shard(sess, input_base_dir, image_filenames, output_filename):
    """Create a TFRecord file from a list of image filenames"""
    writer = tf.python_io.TFRecordWriter(output_filename)
    
    for filename in image_filenames:
        path_filename = os.path.join(input_base_dir,filename)
        #print('wave file: ', path_filename)
        
        #if os.stat(path_filename).st_size == 0:
        #    print('SKIPPING',filename)
        #    continue
        #try:
        
        image_filename = os.path.splitext(path_filename)[0]+'.jpg'
        print('looking for file: ',image_filename)

        #If ty
        

        #print('open: ',path_filename)
        soundfile_obj2 = sf.SoundFile(path_filename)
        #print('open: ',path_filename,soundfile_obj2.subtype)
        if soundfile_obj2.subtype == 'FLOAT' or soundfile_obj2.subtype == 'IMA_ADPCM' or soundfile_obj2.subtype=='MS_ADPCM':
            print('skip because of subtype FLOAT or IMA_ADPCM or MS_ADPCM, file: ', path_filename)
            #input('skip')
            continue
        soundfile_obj2.close()
        

        if os.path.isfile(image_filename) and resave_waves_as_jpgs==False:
            print('skipping jpg file', image_filename)
        else:
            #print('saving: ',image_filename)
            #input('saving')
            image_data_temp, height, width = log_scale_melspectrogram(sess,path_filename,False)
            scipy.misc.imsave(image_filename, image_data_temp)

        image_data,height,width = get_image(sess,image_filename)
        #image_data = bytearray(image_data_temp)
        #print(type(image_data))

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


def get_image_filenames(image_list_filename):
    """ Given input file, generate a list of relative filenames"""
    filenames = []
    with open(image_list_filename) as f:
        for line in f:
            # Carve out the ground truth string and file path from lines like:
            # ./2697/6/466_MONIKER_49537.jpg 49537
            filename = line.split(' ',1)[0][2:] # split off "./" and number
            filenames.append(filename)
    return filenames

def get_image(sess,filename):
    """Given path to an image file, load its data and size"""
    with tf.gfile.FastGFile(filename, 'rb') as f: # added 'b' python3
        image_data = f.read()

    image = sess.run(jpeg_decoder,feed_dict={jpeg_data: image_data})
    height = image.shape[0]
    width = image.shape[1]

    return image_data, height, width

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
    text = os.path.basename(filename).split('-',2)[1]
    #print(text)
    #input('after')
    #OLD
    #text = os.path.basename(filename).split('_',2)[1]
    # Transform string text to sequence of indices using charset, e.g.,
    # MONIKER -> [12, 14, 13, 8, 10, 4, 17]
    #NEW
    labels = [out_charset.index(c) for c in list(text)]
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
    labels = labels+[6]+[6]+[6]+[6]+[6]+[6]+[6]+[6]+[6]
    print('lables',labels)
    print('width:',width)
    print('height:',height)
    print('filename',filename)
    text = text + "666666666"
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
    
    gen_data('../../UrbanSound8K/audio','../../UrbanSound8K/metadata/UrbanSound8K.csv','../data/melspectrums/mel')
    #gen_data('../data/images', 'annotation_val.txt',   '../data/val/words')
    #gen_data('../data/images', 'annotation_test.txt',  '../data/test/words')

if __name__ == '__main__':
    main()
