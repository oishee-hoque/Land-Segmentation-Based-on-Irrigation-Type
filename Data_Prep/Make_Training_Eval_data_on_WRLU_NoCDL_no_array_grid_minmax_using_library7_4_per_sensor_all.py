'''
#### This one normalizes based on NLCD ag areas average and standard deviation
'''


#!/usr/bin/env python
# coding: utf-8

# # Tensorflow and earthengine at work

# ### Import tensorflow and check version

# In[1]:


import os


# ### Import ee and folium

# In[2]:


import ee

# Initialize the Earth Engine module.
ee.Initialize()


# In[3]:


# Folium setup.
import folium
print(folium.__version__)


# # Functions

# In[64]:


IRRI_METHOD = {'Flood':0, 'F':0, 'S':1, 'Sprinkler':1}
##Drip irrigation dropped because very limited

IRRI_METHOD_list = [key for key in IRRI_METHOD]

RESPONSE = ['flood', 'sprinkler', 'other']

def filter_by_property(feat_coll, propert, values):
    expression_ = ''
    for val in values:
        expression_ += '{0} == \"{1}\" || '.format(propert, val)
    expression_ = expression_.strip(' || ')
    expression_ += ''
    filt = ee.Filter.expression(expression_)
    new_feats = feat_coll.filter(filt)
    return new_feats


def get_irrig_array(feat):
    global IRRNAME
    feat = ee.Feature(feat)
    thisirrg = ee.String(feat.get(IRRNAME))
    thisdict = ee.Dictionary(IRRI_METHOD)
    thisval = thisdict.get(thisirrg)
    feat = feat.set({'Irrigation':thisval})
    return feat
    
def get_wrlu_dataset(year_):
    assert (year_ >= 2003 and year_ <= 2021), "Year out of range"
    all_features = ee.FeatureCollection([])
    if year_ in range(2003, 2008):
        dname = 'projects/ee-snouwakpo/assets/Water_Related_Land_Use_2002_to_2007'
        all_features = all_features.merge(ee.FeatureCollection(dname).filter(ee.Filter.expression(f'SURV_YEAR == \"{year_}\"')))
     
    if year_ in range(2005,2011):
        dname = 'projects/ee-snouwakpo/assets/Water_Related_Land_Use_2005_to_2010'
        all_features = all_features.merge(ee.FeatureCollection(dname).filter(ee.Filter.expression(f'SURV_YEAR == \"{year_}\"')))
    
    if year_ in range(2010,2016):
        dname = 'projects/ee-snouwakpo/assets/Water_Related_Land_Use_2010_to_2015'
        all_features = all_features.merge(ee.FeatureCollection(dname).filter(ee.Filter.expression(f'SURV_YEAR == {year_}')))
    if year_ >= 2018:
        dname = 'projects/ee-snouwakpo/assets/Water_Related_Land_Use_Statewide_{year_}'
        all_features = ee.FeatureCollection(dname)
    print(all_features.size().getInfo())
    return all_features


# # Variables
# 
# Declare the variables that will be in use throughout the notebook.

# # Some other global variables

# In[292]:


# Specify names locations for outputs in Cloud Storage. 
FOLDER = 'myCNN'
TRAINING_BASE = 'training_patches'
EVAL_BASE = 'eval_patches'



# Sizes of the training and evaluation datasets.
TRAIN_SIZE = 40
EVAL_SIZE = 10

CRS_ = 'EPSG:3857'
SCALE_ = 30
KERNEL_SIZE = 256
KERNEL_DIM = SCALE_*(KERNEL_SIZE-1)/2 ##30m x kernel size (30*(32-1)/2)
print(KERNEL_DIM)

# Specify model training parameters.
BATCH_SIZE = 16
EPOCHS = 10
BUFFER_SIZE = 2000
OPTIMIZER = 'Adam'
LOSS = 'MeanSquaredError'
METRICS = ['RootMeanSquaredError']


FOLIUMLOCATION = [39.811,-111.625,]

#BOUND = ee.Geometry.Polygon([[-114.39600360146763,37.00483311259188],[-108.24365985146763,37.00483311259188],
#         [-108.24365985146763,43.18322928026068], [-114.39600360146763,43.18322928026068],
#         [-114.39600360146763,37.00483311259188]])

BOUND = ee.FeatureCollection("users/snouwakpo/ML_Training_Area").filter("NAME == 'UTAH'").first().geometry().bounds()



# Period to use to grab images
# 0 = Apr-Aug, 1 = Apr-June, 2 = July-Aug
IMPERIOD = 0 
PERIODDATA = [{'folder':'Landsat_April_August','months':[3,8]},
              {'folder':'Landsat_April_June','months':[3,6]},
              {'folder':'Landsat_July_August','months':[6,8]}]
FOLDER = PERIODDATA[IMPERIOD]['folder']
MONTHS = PERIODDATA[IMPERIOD]['months']


# In[293]:

import sys

YEAR_SEL = int(sys.argv[1]) ##Takes year as argument

print(f'Starting job for year {YEAR_SEL}')

TILESCALE = 16
#UTAH_WATER = ee.FeatureCollection('users/snouwakpo/Utah_Water_Related_Land_Use-shp')
UTAH_WATER = get_wrlu_dataset(YEAR_SEL)
print(UTAH_WATER.size().getInfo())
IRRNAME = ee.Feature(UTAH_WATER.first()).select(['IRR_.*']).propertyNames().getInfo()
print(IRRNAME)
assert len(IRRNAME) == 1, "Check irrigation methods column name"
IRRNAME = IRRNAME[0]


# In[294]:


#UTAH_AG_WATER = filter_by_property(UTAH_WATER, 'Landuse', ['Agricultural']) #Select only ag areas
UTAH_AG_WATER = filter_by_property(UTAH_WATER, IRRNAME, IRRI_METHOD_list) #Select only ag areas


# In[295]:


nFields_surveyed = UTAH_AG_WATER.size().getInfo()
print(nFields_surveyed)


# In[296]:


#Create response data
#UTAH_IRR_DICT = UTAH_AG_WATER.map(get_irrig_array, True) ##This to only run on agricultural fields
UTAH_IRR_DICT = UTAH_AG_WATER.map(get_irrig_array, True) ##Run on ag and non-ag features



UTAH_IRR_IMG = UTAH_IRR_DICT.reduceToImage(properties = ['Irrigation'], reducer = ee.Reducer.first()).rename('Irrigation')
#UNIFIED_IRR_BOUND = UTAH_IRR_DICT.union().first().geometry().simplify(maxError= 100)
UNIFIED_IRR_BOUND = UTAH_IRR_IMG.gte(0).selfMask().reduceToVectors(geometry=BOUND, crs=CRS_, scale=SCALE_, bestEffort=True)
FLOOD_IRR_BOUND = UTAH_IRR_IMG.eq(0).reduceToVectors(geometry=BOUND, crs=CRS_, scale=SCALE_, bestEffort=True)
SPRINKLER_IRR_BOUND = UTAH_IRR_IMG.eq(1).reduceToVectors(geometry=BOUND, crs=CRS_, scale=SCALE_, bestEffort=True)

print(UTAH_IRR_DICT.first().getInfo().get('properties').get('Irrigation'))
print(UTAH_IRR_IMG.getInfo())


# In[297]:

training_squares_ = ee.FeatureCollection(f'projects/ee-snouwakpo/assets/Training_Samples_{KERNEL_SIZE}_{YEAR_SEL}')

evaluation_squares_ = ee.FeatureCollection(f'projects/ee-snouwakpo/assets/Evaluation_Samples_{KERNEL_SIZE}_{YEAR_SEL}')




# In[298]:

# In[299]:


MASK = UTAH_IRR_IMG.mask()

##Fill with value 2 for the other pixels
two = ee.Image(2).rename('Irrigation')
UTAH_IRR_IMG = two.where(MASK, UTAH_IRR_IMG).toUint8()


# Turn irrigation image into multiband

# In[300]:


expression_flood = 'b == 0 ? 1 : 0'
expression_sprinkler = 'b == 1 ? 1 : 0'
expression_other = 'b == 2 ? 1 : 0'

flood = ee.Image().expression(expression_flood,
                                      {'b':UTAH_IRR_IMG.select("Irrigation")}).rename(RESPONSE[0])
sprinkler = ee.Image().expression(expression_sprinkler,
                                      {'b':UTAH_IRR_IMG.select("Irrigation")}).rename(RESPONSE[1])
otherirrig = ee.Image().expression(expression_other,
                                      {'b':UTAH_IRR_IMG.select("Irrigation")}).rename(RESPONSE[2])
IRRIGATION_IMG = ee.Image.cat([flood, sprinkler, otherirrig])

other_threshold = KERNEL_SIZE*KERNEL_SIZE*0.9 #At least 10% of the area has to be flood or sprinkler



# # Imagery
# 
# Gather and setup the imagery to use for inputs (predictors).  This is a three-year, cloud-free, Landsat 8 composite.  Display it in the notebook for a sanity check.

# In[301]:



##########################################################################




training_year = YEAR_SEL

import ml_input_maker7_4 as ml_input_maker

thisInputClass = ml_input_maker.input_maker(training_year, BOUND)


AgAreas = thisInputClass.AGAREAS

landsat_col = thisInputClass.landsat_col
#otherbands = thisInputClass.otherbands
inputbands = thisInputClass.inputbands


GOTBANDS=False

while not GOTBANDS:
    try:
        BANDS = inputbands.bandNames().getInfo()
        print(BANDS)
        GOTBANDS = True
    except ee.ee_exception.EEException:
        print('Issue getting Band names from EE')
    except:
        print('Another error occured')
        break

assert GOTBANDS, 'BAND NAMES COULD NOT BE RETRIEVED. STOPPING!!!'






FEATURES = BANDS+RESPONSE
print(FEATURES)



# In[302]:


#print(YEAR_SEL)

# ## Build feature stack

# In[303]:




featureStack = ee.Image.cat([
  inputbands.select(BANDS),
  IRRIGATION_IMG
]).float()

# In[304]:


## Create the arrays now
projected_stack = featureStack.reproject(crs = CRS_, scale=SCALE_)
def create_array_mapping(feat_in):
    centroid = feat_in.centroid().transform(CRS_)
    coords = centroid.geometry().coordinates()
    new_coords = ee.List([[ee.Number(coords.get(0)).add(0-KERNEL_DIM), ee.Number(coords.get(1)).add(0-KERNEL_DIM)], 
                  [ee.Number(coords.get(0)).add(KERNEL_DIM),ee.Number(coords.get(1)).add(KERNEL_DIM)]])
    bbox = ee.Geometry.Rectangle(coords=new_coords, 
                                 proj=CRS_, geodesic=False)
    sampled_arr = projected_stack.sampleRectangle(bbox, defaultValue=0)
    sampled_arr = sampled_arr.copyProperties(feat_in)
    other_count = sampled_arr.getArray('other').reshape([-1]).reduce(reducer = ee.Reducer.sum(), axes=[0]).get([0])
    flood_count = sampled_arr.getArray('flood').reshape([-1]).reduce(reducer = ee.Reducer.sum(), axes=[0]).get([0])
    sprinkler_count = sampled_arr.getArray('sprinkler').reshape([-1]).reduce(reducer = ee.Reducer.sum(), axes=[0]).get([0])
    test_1 = ee.Number(other_count).lte(other_threshold)
    test_2 = ee.Number(other_count).add(ee.Number(flood_count)).add(ee.Number(sprinkler_count))
    test_2 = test_2.neq(0)
    test_ = test_1.And(test_2)
    #test_ = test_2
    
    sampled_arr = ee.Algorithms.If(test_, sampled_arr)
    
    return sampled_arr




training_List = training_squares_.toList(30000)
training_List = training_List.shuffle()
training_List = training_List.slice(0,2000)
training_arrays = ee.FeatureCollection(training_List).map(create_array_mapping, True) #Takes the whole set but divides by 2
evaluation_List = evaluation_squares_.toList(30000)
evaluation_List = evaluation_List.shuffle()
evaluation_List = evaluation_List.slice(0,500)
evaluation_arrays = ee.FeatureCollection(evaluation_List).map(create_array_mapping, True) #Takes the whole set but divides by 2

#training_vals = training_arrays.first().getInfo()

import numpy as np

#img_size = np.array(training_vals['properties']['other'])
#print(f'IMGSIZE = {img_size.shape}')
#print([training_attr for training_attr in training_vals])

# ## Functions to save and load data as TFRecords

# In[279]:





try:
    ntraining = training_arrays.size().getInfo()
    neval = evaluation_arrays.size().getInfo()
except:
    ntraining = 1000
    neval = 250




def sample_AOC_to_tfrecord_squares(fname):
    desc1 = fname + '_g_train'
    if ntraining > 0:
        print(f'Saving training samples for {YEAR_SEL}')
        task = ee.batch.Export.table.toDrive(
            collection = training_arrays,
            description = desc1,
            folder = f'Irrigation_detection_WRLU_NoCDL_Balanced_Normalized_Unique_UT_lib7_4',
            fileFormat = 'TFRecord',
            selectors = FEATURES
        )
        task.start()
    desc2 = fname + '_g_eval'
    if neval > 0:
        print(f'Saving evaluation samples for {YEAR_SEL}')
        task = ee.batch.Export.table.toDrive(
            collection = evaluation_arrays,
            description = desc2,
            folder = f'Irrigation_detection_WRLU_NoCDL_Balanced_Normalized_Unique_UT_lib7_4',
            fileFormat = 'TFRecord',
            selectors = FEATURES
        )
        task.start()
        #TASKS.append(task)
    return


# ### Write training and evaluation sets

# In[280]:



sample_AOC_to_tfrecord_squares(f'Samples_{YEAR_SEL}_{KERNEL_SIZE}x{KERNEL_SIZE}')




# In[ ]:



