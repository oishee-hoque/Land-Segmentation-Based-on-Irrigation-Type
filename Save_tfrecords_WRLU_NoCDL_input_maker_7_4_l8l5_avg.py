#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
print(tf.__version__)


# In[34]:



# Sizes of the training and evaluation datasets.
#TRAIN_SIZE = 1703 #Use when non-ag included (does not include slope)
## Approximate total number of 32x32 areas extracted:
TOTSAMPLE = 3*(1539+1539+1569+261+114+49+105+486+177+147+203+275+282+197) #New balanced dataset is 3times as large as original 
                                                                          #N squares = number of fields /30 instead of /100
## 80% of that is training


#TRAIN_SIZE = int(TOTSAMPLE*0.8)
#TRAIN_SIZE = 16231
#TRAIN_SIZE = 3280 #Unique
#TRAIN_SIZE = 633 #Unique AOC
#TRAIN_SIZE = 894 #Unique AOC2
#TRAIN_SIZE = 682 #Unique AOC3
#TRAIN_SIZE = 1718 #Unique UT_l8_1
#TRAIN_SIZE = 2277 #Unique UT_l8_8_29_22
TRAIN_SIZE = 3929 #Unique UT_l8
#EVAL_SIZE = 395 #Use when non-ag included (does not include slope)
#EVAL_SIZE = TOTSAMPLE - TRAIN_SIZE
#EVAL_SIZE = 4100
#EVAL_SIZE = 824
#EVAL_SIZE = 144 #Unique AOC
#EVAL_SIZE = 246 #Unique AOC2
#EVAL_SIZE = 187 #Unique AOC3
#EVAL_SIZE = 383 #Unique UT_l8_1
#EVAL_SIZE = 242 #Unique UT_l8_2
#EVAL_SIZE = 498 #Unique UT_l8_8_29_22
EVAL_SIZE = 876 #Unique UT_l8

# Specify model training parameters.
BATCH_SIZE = 64
EPOCHS = 1000 #Should auto-stop
BUFFER_SIZE = 2*(TRAIN_SIZE+BATCH_SIZE)
print(f'Estimated parameters TOTSAMPLE={TOTSAMPLE},TRAIN_SIZE={TRAIN_SIZE},EVAL_SIZE={EVAL_SIZE}, BUFFER_SIZE={BUFFER_SIZE}')


YEARS_PYTHON = ['2005', '2016']
FOLIUMLOCATION = [39.811,-111.625,]


ALL_BANDS = ['0_BGR0_median', '0_BGR1_median', '0_BGR2_median', '0_SWIR0_median', '0_SWIR1_median', '0_SWIR2_median', '0_SR_TH_median', 
             '0_BGR0_diff', '0_BGR1_diff', '0_BGR2_diff', '0_SWIR0_diff', '0_SWIR1_diff', '0_SWIR2_diff', '0_SR_TH_diff', '1_BGR0_median', 
             '1_BGR1_median', '1_BGR2_median', '1_SWIR0_median', '1_SWIR1_median', '1_SWIR2_median', '1_SR_TH_median', '1_BGR0_diff', 
             '1_BGR1_diff', '1_BGR2_diff', '1_SWIR0_diff', '1_SWIR1_diff', '1_SWIR2_diff', '1_SR_TH_diff', '2_BGR0_median', '2_BGR1_median', 
             '2_BGR2_median', '2_SWIR0_median', '2_SWIR1_median', '2_SWIR2_median', '2_SR_TH_median', '2_BGR0_diff', '2_BGR1_diff', '2_BGR2_diff', 
             '2_SWIR0_diff', '2_SWIR1_diff', '2_SWIR2_diff', '2_SR_TH_diff', '3_BGR0_median', '3_BGR1_median', '3_BGR2_median', '3_SWIR0_median', 
             '3_SWIR1_median', '3_SWIR2_median', '3_SR_TH_median', '3_BGR0_diff', '3_BGR1_diff', '3_BGR2_diff', '3_SWIR0_diff', '3_SWIR1_diff', 
             '3_SWIR2_diff', '3_SR_TH_diff', 'BGR0_stdDev', 'BGR1_stdDev', 'BGR2_stdDev', 'SWIR0_stdDev', 'SWIR1_stdDev', 'SWIR2_stdDev', 'SR_TH_stdDev']



SAVED_BANDS = [f'BGR{i}_median' for i in range(3)]+[f'SWIR{i}_median' for i in range(3)]+[f'SR_TH_median']+\
                [f'BGR{i}_diff' for i in range(3)]+[f'SWIR{i}_diff' for i in range(3)]+[f'SR_TH_diff']+\
                [f'BGR{i}_stdDev' for i in range(3)]+[f'SWIR{i}_stdDev' for i in range(3)]+[f'SR_TH_stdDev']

RESPONSE = ['flood', 'sprinkler', 'other'] 

ALL_FEATURES = ALL_BANDS+RESPONSE



# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = 64
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
COLUMNS = [
  tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in ALL_FEATURES
]
COLUMNS2 = [
  tf.io.FixedLenFeature([], tf.string) for k in ALL_FEATURES
]
ALL_FEATURES_DICT = dict(zip(ALL_FEATURES, COLUMNS))
ALL_FEATURES_DICT_STRING = dict(zip(ALL_FEATURES, COLUMNS2))




# Specify names locations for outputs in Cloud Storage. 
BASEFOLDER = '/' + 'project/nwisrl_water/nwisrl_water' + '/' + 'Irrigation_detection' + '/' + 'Irrigation_detection_lib7_4'
TRAINING_BASE = f'{KERNEL_SIZE}x{KERNEL_SIZE}_g_train'
EVAL_BASE = f'{KERNEL_SIZE}x{KERNEL_SIZE}_g_eval'



# # Training data
# 
# Load the data exported from Earth Engine into a `tf.data.Dataset`.  The following are helper functions for that.

# In[35]:

import sys
from tensorflow.keras import layers

def parse_tfrecord(example_proto):
  """The parsing function.
  Read a serialized example into the structure defined by FEATURES_DICT.
  Args:
    example_proto: a serialized Example.
  Returns:
    A dictionary of tensors, keyed by feature name.
  """
  return tf.io.parse_single_example(example_proto, ALL_FEATURES_DICT)

def to_tuple(inputs):
  """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
  Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
  Args:
    inputs: A dictionary of tensors, keyed by feature name.
  Returns:
    A tuple of (inputs, outputs).
  """
  inputsList = [inputs.get(key) for key in ALL_FEATURES]
  
  stacked = tf.stack(inputsList, axis=0)
  # Convert from CHW to HWC
  stacked = tf.transpose(stacked, [1, 2, 0])
  return stacked

def sample_to_dict(inputs):
    """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
    inputs: A dictionary of tensors, keyed by feature name.
    Returns:
    A tuple of (inputs, outputs).
    """
    inputsList = {key:inputs.get(key) for key in ALL_FEATURES}
    return inputsList


def normalize_input(sample_dict):
    '''This augments the data by applying random filps 
    and adding random noise
    '''
    for key in ALL_FEATURES:
        if not (key in RESPONSE):
            if True:
                mean_ = tf.math.reduce_mean(sample_dict[key])
                std_ = tf.math.reduce_std(sample_dict[key])
                std_ = tf.math.add(std_, 1e-5)
                centered = tf.math.subtract(sample_dict[key], mean_)
                normalized_ = tf.math.divide(centered, std_)
                normalized_ = tf.math.divide(normalized_, 3.0)
                sample_dict[key] = normalized_
            else:
                sample_dict[key] = tf.math.divide(sample_dict[key], 65536)
    return sample_dict

def save_average_input(sample_dict):
    '''This augments the data by applying random filps 
    and adding random noise
    '''
    to_drop = []
    sample_dict_out = {}
    for band_ in ['BGR', 'SWIR']:
        for i in range(3):
            to_avg = [sample_dict[f'{j}_{band_}{i}_median'] for j in range(4)]
            to_drop += [f'{j}_{band_}{i}_median' for j in range(4)]
            accumulate_vals = tf.math.add_n(to_avg)
            mean_vals = tf.math.divide(accumulate_vals, float(len(to_avg)))
            sample_dict_out[f'{band_}{i}_median'] = mean_vals
        for i in range(3):
            to_avg = [sample_dict[f'{j}_{band_}{i}_diff'] for j in range(4)]
            to_drop += [f'{j}_{band_}{i}_diff' for j in range(4)]
            accumulate_vals = tf.math.add_n(to_avg)
            mean_vals = tf.math.divide(accumulate_vals, float(len(to_avg)))
            sample_dict_out[f'{band_}{i}_diff'] = mean_vals
            
    for band_ in ['SR_TH']:
        to_avg = [sample_dict[f'{j}_{band_}_median'] for j in range(4)]
        to_drop += [f'{j}_{band_}_median' for j in range(4)]
        accumulate_vals = tf.math.add_n(to_avg)
        mean_vals = tf.math.divide(accumulate_vals, float(len(to_avg)))
        sample_dict_out[f'{band_}_median'] = mean_vals
        
        to_avg = [sample_dict[f'{j}_{band_}_diff'] for j in range(4)]
        to_drop += [f'{j}_{band_}_diff' for j in range(4)]
        accumulate_vals = tf.math.add_n(to_avg)
        mean_vals = tf.math.divide(accumulate_vals, float(len(to_avg)))
        sample_dict_out[f'{band_}_diff'] = mean_vals
    for key in ALL_FEATURES:
        if key in to_drop:
            continue
        sample_dict_out[key] = sample_dict[key]
    return sample_dict_out

def check_other_class(sample_dict):
    '''
    This checks that the class other is not more than a certain threshold
    This is to avoid having an over representation of non-ag pixels
    '''
    other_threshold = 0.66*KERNEL_SIZE*KERNEL_SIZE
    sum_other = tf.math.reduce_sum(sample_dict['other'])
    if tf.math.greater(sum_other, other_threshold):
        return False
    else:
        return True
        


class tfr_fhandle:
    def __init__(self, fname):
        filename = fname+".tfrecords"
        self.tfrwriter = tf.io.TFRecordWriter(filename)
    def close(self):
        self.tfrwriter.close()
    def write(self, datastream):
        self.tfrwriter.write(datastream)
        
        

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

import numpy as np

def parse_single_sample(feature):
    data = {}
    for feature_name in SAVED_BANDS:
        data[feature_name] = _bytes_feature(serialize_array(np.array(feature[feature_name], dtype = 'float32')))
    for resp in RESPONSE:
        data[resp] = _bytes_feature(serialize_array(np.array(feature[resp], dtype = 'float32')))
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out


def write_examples_to_tfr(all_features_list, tfr_writer):
    count_ = 0
    for thisfeature in all_features_list:
        if not check_other_class(thisfeature):
            continue
        out = parse_single_sample(thisfeature)
        tfr_writer.write(out.SerializeToString()) ##Uncomment this if needed to write to the one aggregated tfrecord file
        count_ += 1
    print(f"Wrote {count_} elements to TFRecord")
    return count_


def get_dataset(fname):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
    pattern: A file pattern to match in a Cloud Storage bucket.
    Returns:
    A tf.data.Dataset
    """
    print(fname)
    dataset = tf.data.TFRecordDataset(fname, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(sample_to_dict)
    dataset = dataset.map(normalize_input)
    dataset = dataset.map(save_average_input)
    return dataset


# Use the helpers to read in the training dataset.  Print the first record to check.

# In[36]:
YEARS = []

MINYEAR = 2003
MAXYEAR = 2022

FOLDERS = {}

for year in range(MINYEAR,MAXYEAR):
    if year not in [2012,2016,2017]:
        YEARS.append(year)
    FOLDERS[year]=BASEFOLDER+'_l8l5_avg_2'
        
        
NSamples = {BASEFOLDER+'_l8l5_avg_2':{'Training':{},'Evaluation':{}}}


def save_training_dataset():
    """Get the preprocessed training dataset
    Returns: 
    A tf.data.Dataset of training data.
    """
    
    for year in YEARS:
        FOLDER = FOLDERS[year]
        fname = f'{FOLDER}/Samples_{year}_{TRAINING_BASE}.tfrecord.gz'
        dataset = get_dataset(fname)
        dat_fname = f'{FOLDER}/Proc_Samples_{year}_{TRAINING_BASE}'
        tfr_writer = tfr_fhandle(dat_fname)
        ncount = write_examples_to_tfr(dataset, tfr_writer)
        NSamples[FOLDER]['Training'][year]=ncount
        if os.path.exists(f'{FOLDER}/Samples_TFCC_{year}_{TRAINING_BASE}.tfrecord.gz'):
            fname = f'{FOLDER}/Samples_TFCC_{year}_{TRAINING_BASE}.tfrecord.gz'
            dataset = get_dataset(fname)
            ncount = write_examples_to_tfr(dataset, tfr_writer)
            NSamples[FOLDER]['Training'][year] += ncount
        tfr_writer.close()
        
    return 

def save_eval_dataset():
    """Get the preprocessed training dataset
    Returns: 
    A tf.data.Dataset of training data.
    """
    for year in YEARS:
        FOLDER = FOLDERS[year]
        fname = f'{FOLDER}/Samples_{year}_{EVAL_BASE}.tfrecord.gz'
        dataset = get_dataset(fname)
        dat_fname = f'{FOLDER}/Proc_Samples_{year}_{EVAL_BASE}'
        tfr_writer = tfr_fhandle(dat_fname)
        ncount = write_examples_to_tfr(dataset, tfr_writer)
        NSamples[FOLDER]['Evaluation'][year]=ncount
        if os.path.exists(f'{FOLDER}/Samples_TFCC_{year}_{EVAL_BASE}.tfrecord.gz'):
            fname = f'{FOLDER}/Samples_TFCC_{year}_{EVAL_BASE}.tfrecord.gz'
            dataset = get_dataset(fname)
            ncount = write_examples_to_tfr(dataset, tfr_writer)
            NSamples[FOLDER]['Evaluation'][year] += ncount
        
        tfr_writer.close()
    return 

save_training_dataset()
save_eval_dataset()
for fold_ in NSamples:
    outputjson = f'{fold_}/NSamples.json'
    import json
    with open(outputjson,'w') as fh:
        json.dump(NSamples[fold_], fh)


